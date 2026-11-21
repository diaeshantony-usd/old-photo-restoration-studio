"""Enhancement stage: Real-ESRGAN (whole image) + GFPGAN (faces only), Wang et al. (2021).

Develop and explain this stage in ``notebooks/05_enhancement``; the code lives here so the
notebook, the tests and the Gradio demo all run the same implementation. Model packages are
imported lazily, so the helper functions can be imported and tested without them.
"""

from __future__ import annotations

import logging
import os
import subprocess

import cv2
import numpy as np
import torch

log = logging.getLogger("enhancer")

MAX_OUTPUT_SIDE = 4096     # longest output side, in pixels
MIN_FACE_PX = 32           # faces smaller than this (on the input) are skipped

WEIGHT_URLS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
}


# --------------------------------------------------------------------------- helpers
def compute_outscale(h: int, w: int, scale: float = 4.0, cap: int = MAX_OUTPUT_SIDE) -> float:
    """Requested upscale, reduced so max(h, w) * outscale <= cap. Never below 1."""
    return float(max(1.0, min(scale, cap / max(h, w))))


def keep_face(bbox, min_size: int = MIN_FACE_PX) -> bool:
    """bbox = [x1, y1, x2, y2, (score)] in input-image pixels."""
    return min(bbox[2] - bbox[0], bbox[3] - bbox[1]) >= min_size


# --------------------------------------------------------------------------- models
def load_models(model_dir: str = "models", tile: int = 512, half: bool = True, download: bool = True):
    """Return (RealESRGANer, GFPGANer). Weights are downloaded into ``model_dir`` (not committed)."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer

    os.makedirs(model_dir, exist_ok=True)
    if download:
        for name, url in WEIGHT_URLS.items():
            path = os.path.join(model_dir, name)
            if not os.path.exists(path):
                subprocess.run(["wget", "-q", "-O", path, url], check=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=os.path.join(model_dir, "RealESRGAN_x4plus.pth"), model=rrdb,
                             tile=tile, tile_pad=10, pre_pad=0, half=half and device == "cuda", device=device)
    face = GFPGANer(model_path=os.path.join(model_dir, "GFPGANv1.4.pth"), upscale=4, arch="clean",
                    channel_multiplier=2, bg_upsampler=None, device=device)
    return upsampler, face


class Enhancer:
    """Real-ESRGAN whole-image upscale + GFPGAN on detected faces.

    * Output is capped at 4096 px on the longest side.
    * Faces under 32 px are skipped; with no usable face only Real-ESRGAN runs.
    * Each restored face is blended with the unrestored face at ``face_weight``
      (1.0 = fully restored, 0.0 = untouched) to limit identity drift.
    * On GPU out-of-memory the tile size is halved and the run retried.

    ``last_info`` records faces found, restored and skipped, the scale used and the face step.
    """

    def __init__(self, upsampler, gfpgan, scale: float = 4.0, face_weight: float = 0.7,
                 enabled: bool = True, restore_faces: bool = True):
        self.up, self.g = upsampler, gfpgan
        self.scale, self.face_weight = scale, face_weight
        self.enabled, self.restore_faces = enabled, restore_faces
        self.last_info: dict = {}

    def _upscale(self, bgr: np.ndarray, outscale: float) -> np.ndarray:
        while True:
            try:
                return self.up.enhance(bgr, outscale=outscale)[0]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_tile = (self.up.tile or 512) // 2
                if new_tile < 64:
                    raise
                log.warning("out of GPU memory: tile %s -> %s", self.up.tile, new_tile)
                self.up.tile = new_tile

    @torch.inference_mode()
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """image: H x W x 3 uint8 RGB. Returns uint8 RGB, up to ``scale`` times larger."""
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        h, w = image.shape[:2]
        if not self.enabled:
            self.last_info = dict(ran=False, reason="enhancement switched off")
            return image
        outscale = compute_outscale(h, w, self.scale)
        info = dict(ran=True, outscale=outscale, faces_found=0, faces_restored=0, faces_too_small=0,
                    face_step="skipped (face restoration off)" if not self.restore_faces else "")

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out = self._upscale(bgr, outscale)

        if self.restore_faces:
            fh = self.g.face_helper
            device = next(self.g.gfpgan.parameters()).device
            fh.clean_all()
            fh.read_image(bgr)
            fh.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
            info["faces_found"] = len(fh.det_faces)

            keep = [i for i, b in enumerate(fh.det_faces) if keep_face(b)]
            info["faces_too_small"] = len(fh.det_faces) - len(keep)
            fh.det_faces = [fh.det_faces[i] for i in keep]
            fh.all_landmarks_5 = [fh.all_landmarks_5[i] for i in keep]

            if not keep:
                info["face_step"] = ("skipped (no face found)" if info["faces_found"] == 0
                                     else "skipped (faces under 32 px)")
            else:
                fh.align_warp_face()
                for crop in fh.cropped_faces:
                    t = img2tensor(crop / 255.0, bgr2rgb=True, float32=True)
                    normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    t = t.unsqueeze(0).to(device)
                    try:
                        restored = tensor2img(self.g.gfpgan(t, return_rgb=False, weight=0.5)[0].squeeze(0),
                                              rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
                    except RuntimeError:                  # GFPGAN failed on this crop: keep the original
                        restored = crop
                    fh.add_restored_face(cv2.addWeighted(restored, self.face_weight, crop, 1 - self.face_weight, 0))
                fh.get_inverse_affine(None)
                out = fh.paste_faces_to_input_image(upsample_img=out)
                info["faces_restored"] = len(keep)
                info["face_step"] = f"restored {len(keep)} face(s)"
            fh.clean_all()

        log.info("%s", info)
        self.last_info = info
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
