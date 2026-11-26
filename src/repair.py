"""Photo repair (notebook 03): classical OpenCV repair and Wan et al. (2020) latent-space restoration.

Methods provided:

  Classical repair: Telea fast-marching inpainting on the damage mask, then Non-Local Means
      denoising and CLAHE contrast enhancement. Because denoising and CLAHE change every pixel,
      a masked-only variant (Telea inside the mask, everything else untouched) is also
      provided so methods can be compared like for like.
  Pretrained Wan et al.: Microsoft's published "Bringing Old Photos Back to Life" scratch
      model (VAE_A_quality + VAE_B_scratch + mapping_scratch with the partial non-local
      block), run unchanged on our photos with our damage masks.
  Fine-tuned Wan et al.: the same published model with both VAEs frozen and only the
      latent mapping network fine-tuned on OpenPhoto Restore.

The learned tone correction and the full repair pipeline live in ``src/tone.py``.

Every restorer follows the shared interface::

    restore(image: np.ndarray, mask: np.ndarray) -> np.ndarray   # H x W x 3 uint8 RGB

The Wan et al. code and checkpoints are Microsoft's (MIT licence) and are downloaded at run
time by :func:`setup_wan_repo`; nothing from that repository is vendored here.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
LARGE_MISSING_THRESHOLD = 0.20  # more than 20% of pixels masked => repair flagged as low confidence

WAN_REPO_URL = "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git"
WAN_CKPT_URL = (
    "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/"
    "global_checkpoints.zip"
)
DEFAULT_FINETUNED_CKPT = ROOT_DIR / "models" / "repair_wan_mapping_finetuned.pt"


def _info(mode: str, mask_ratio: float, note: str) -> Dict[str, object]:
    low = mask_ratio > LARGE_MISSING_THRESHOLD
    if low:
        note = "warning: large missing region (>20% masked), low confidence"
    return {"mode": mode, "mask_ratio": mask_ratio, "low_confidence": low, "note": note}


# ==============================================================================
# Classical OpenCV repair
# ==============================================================================


class OpenCVBaselineRestorer:
    """Telea inpainting (+ optional NLM denoising and CLAHE).

    ``mask_only=True`` keeps every pixel outside the mask exactly as it was (Telea only),
    which is the fair counterpart to a method that only edits masked pixels.
    """

    def __init__(
        self,
        inpaint_radius: int = 3,
        denoise_h: float = 5.0,
        clahe_clip: float = 1.8,
        mask_only: bool = False,
    ):
        self.inpaint_radius = inpaint_radius
        self.denoise_h = denoise_h
        self.clahe_clip = clahe_clip
        self.mask_only = mask_only
        self.mode = "opencv_telea_mask_only" if mask_only else "opencv_telea_nlm_clahe"
        self.last_info: Dict[str, object] = {}

    def restore(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        mask_u8 = np.zeros(image.shape[:2], np.uint8) if mask is None else (mask > 0).astype(np.uint8) * 255
        ratio = float(np.mean(mask_u8 > 0))
        if ratio == 0.0:
            self.last_info = _info(self.mode, 0.0, "no damage detected; original image preserved")
            return image.copy()

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, mask_u8, inpaintRadius=self.inpaint_radius, flags=cv2.INPAINT_TELEA)
        if not self.mask_only:
            out = cv2.fastNlMeansDenoisingColored(out, None, self.denoise_h, self.denoise_h, 7, 21)
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        if self.mask_only:
            keep = (mask_u8 == 0)[..., None]
            out = np.where(keep, image, out)
        self.last_info = _info(self.mode, ratio, "OpenCV repair completed")
        return out


# ==============================================================================
# Wan et al. (2020): published code + checkpoints
# ==============================================================================


def setup_wan_repo(dest: Path) -> Path:
    """Clones Microsoft's repository and downloads the Global (stage-1) checkpoints once.

    Returns the path of the ``Global`` folder.
    """
    dest = Path(dest)
    if not (dest / "Global").exists():
        subprocess.run(["git", "clone", "--depth", "1", WAN_REPO_URL, str(dest)], check=True)
    global_dir = dest / "Global"
    ckpt_dir = global_dir / "checkpoints" / "restoration"
    if not (ckpt_dir / "mapping_scratch").exists():
        zip_path = global_dir / "global_checkpoints.zip"
        if not zip_path.exists():
            torch.hub.download_url_to_file(WAN_CKPT_URL, str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(global_dir)
    for sub in ("VAE_A_quality", "VAE_B_scratch", "mapping_scratch"):
        assert (ckpt_dir / sub).exists(), f"missing Wan checkpoint folder {ckpt_dir / sub}"
    return global_dir


def load_wan_scratch_model(global_dir: Path, device: Optional[torch.device] = None):
    """Builds Microsoft's ``Pix2PixHDModel_Mapping`` with the published scratch weights.

    Uses exactly the option set of ``Global/test.py --Scratch_and_Quality_restore``. Both
    VAEs come back frozen (``requires_grad=False``, eval mode), as in the original code.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_dir = Path(global_dir).resolve()
    if str(global_dir) not in sys.path:
        sys.path.insert(0, str(global_dir))

    from options.test_options import TestOptions  # noqa: E402  (Microsoft code)
    from models.mapping_model import Pix2PixHDModel_Mapping  # noqa: E402

    gpu = "0" if device.type == "cuda" else "-1"
    old_argv, old_cwd = sys.argv, os.getcwd()
    sys.argv = ["wan", "--Scratch_and_Quality_restore", "--gpu_ids", gpu]
    os.chdir(global_dir)  # the original code resolves ./checkpoints relative to Global/
    try:
        opt = TestOptions().parse(save=False)
        # parameter_set() from Global/test.py, scratch branch
        opt.serial_batches = True
        opt.no_flip = True
        opt.label_nc = 0
        opt.n_downsample_global = 3
        opt.mc = 64
        opt.k_size = 4
        opt.start_r = 1
        opt.mapping_n_block = 6
        opt.map_mc = 512
        opt.no_instance = True
        opt.checkpoints_dir = "./checkpoints/restoration"
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")

        model = Pix2PixHDModel_Mapping()
        model.initialize(opt)
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    model.eval()
    for net in (model.netG_A, model.netG_B):
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
    return model


def wan_forward(model, damaged: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """One pass of the published network on a batch.

    Args:
        damaged: (B, 3, H, W) in [0, 1]; H and W multiples of 8.
        mask:    (B, 1, H, W) in {0, 1}, 1 = damaged.
    Returns dict with ``out`` (B, 3, H, W) in [0, 1] and the mapped latent ``z_y``.

    Mirrors ``Global/test.py``: masked pixels are set to white before encoding
    (``irregular_hole_synthesize``), images are normalised to [-1, 1], VAE_A encodes,
    the mapping network (with the partial non-local block) translates, VAE_B decodes.
    """
    x = damaged * (1.0 - mask) + mask  # white holes, as in irregular_hole_synthesize
    x = x * 2.0 - 1.0
    z_x = model.netG_A.forward(x, flow="enc")
    z_y = model.mapping_net(z_x.detach(), mask)
    out = model.netG_B.forward(z_y, flow="dec")
    return {"out": ((out + 1.0) / 2.0).clamp(0.0, 1.0), "z_y": z_y}


class WanRestorer:
    """Restorer around the published Wan et al. scratch model (pretrained or fine-tuned).

    The network regenerates the whole photo (scratch filling plus global quality
    restoration), so its output differs from the input everywhere; compare it with the
    full OpenCV baseline on whole-image metrics and with every method on masked PSNR.
    """

    def __init__(self, model, mode: str, device: Optional[torch.device] = None, skip_if_clean: bool = True):
        self.model = model
        self.mode = mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_if_clean = skip_if_clean
        self.last_info: Dict[str, object] = {}

    @torch.no_grad()
    def restore(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        m = np.zeros(image.shape[:2], np.float32) if mask is None else (mask > 0).astype(np.float32)
        ratio = float(m.mean())
        if ratio == 0.0 and self.skip_if_clean:
            self.last_info = _info(self.mode, 0.0, "no damage detected; original image preserved")
            return image.copy()

        h, w = image.shape[:2]
        ph, pw = (-h) % 8, (-w) % 8
        img = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        m_p = np.pad(m, ((0, ph), (0, pw)), mode="reflect")
        t_img = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255.0).to(self.device)
        t_msk = torch.from_numpy(m_p)[None, None].to(self.device)
        self.model.mapping_net.eval()
        out = wan_forward(self.model, t_img, t_msk)["out"][0, :, :h, :w]
        out_u8 = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        self.last_info = _info(self.mode, ratio, f"Wan et al. ({self.mode}) restoration completed")
        return out_u8


def finetune_wan_mapping(
    model,
    train_loader,
    epochs: int,
    seed: int,
    lr: float = 1e-4,
    feat_weight: float = 60.0,
    image_weight: float = 10.0,
    val_fn: Optional[Callable[[], float]] = None,
    device: Optional[torch.device] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Fine-tunes only ``model.mapping_net``; VAE_A and VAE_B stay frozen.

    Loss follows the original mapping training (``train_mapping.py``): L1 between the
    mapped latent and VAE_B's encoding of the clean target, weighted by ``l2_feat=60``
    with ``--use_l1_feat``, plus an image L1 (``--Smooth_L1 --image_L1``, weight 10).
    The adversarial and VGG terms of the original recipe are dropped to fit the
    course compute budget; this is stated in the notebook.

    ``val_fn`` (higher is better, e.g. validation masked PSNR) picks the best epoch;
    the best mapping weights are loaded back into ``model`` before returning.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _random

    _random.seed(seed)

    for net in (model.netG_A, model.netG_B):
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
    mapping = model.mapping_net
    for p in mapping.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(mapping.parameters(), lr=lr, betas=(0.5, 0.999))

    history: List[Dict[str, float]] = []
    best_score, best_state = -float("inf"), copy.deepcopy(mapping.state_dict())
    if val_fn is not None:
        mapping.eval()
        best_score = float(val_fn())
        log(f"  [seed {seed}] epoch 0 (published weights) | val {best_score:.3f}")

    for epoch in range(1, epochs + 1):
        mapping.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in train_loader:
            dmg = batch["damaged"].to(device, non_blocking=True)
            tgt = batch["clean"].to(device, non_blocking=True)
            msk = batch["mask"].to(device, non_blocking=True)
            res = wan_forward(model, dmg, msk)
            with torch.no_grad():
                z_target = model.netG_B.forward(tgt * 2.0 - 1.0, flow="enc")
            loss = feat_weight * F.l1_loss(res["z_y"], z_target) + image_weight * F.l1_loss(res["out"], tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapping.parameters(), 5.0)
            opt.step()
            tot += float(loss.item())
            n += 1
        rec = {"epoch": epoch, "train_loss": tot / max(n, 1), "seconds": time.time() - t0}
        if val_fn is not None:
            mapping.eval()
            rec["val"] = float(val_fn())
            if rec["val"] > best_score:
                best_score, best_state = rec["val"], copy.deepcopy(mapping.state_dict())
        else:
            best_state = copy.deepcopy(mapping.state_dict())
        history.append(rec)
        log(
            f"  [seed {seed}] epoch {epoch}/{epochs} | train loss {rec['train_loss']:.4f}"
            + (f" | val {rec['val']:.3f}" if "val" in rec else "")
            + f" | {rec['seconds']:.0f}s"
        )

    mapping.load_state_dict(best_state)
    mapping.eval()
    return {"history": history, "best_val": best_score, "state_dict": best_state}


def _shard_paths(path: Path) -> List[Path]:
    path = Path(path)
    return sorted(path.parent.glob(f"{path.stem}.part*{path.suffix}"))


def save_mapping_weights(
    state_dict: Dict[str, torch.Tensor], path: Path, meta: Optional[Dict] = None, max_shard_mb: float = 90.0
) -> List[Path]:
    """Saves fine-tuned mapping weights in fp16, split into shards below GitHub's 100 MB limit.

    The mapping network has ~51M parameters (~103 MB in fp16), so it is written as
    ``<stem>.part1of2.pt``, ``<stem>.part2of2.pt``; :func:`load_mapping_weights` joins them.
    """
    path = Path(path)
    for old in _shard_paths(path):
        old.unlink()
    half = {k: (v.half() if v.is_floating_point() else v).cpu() for k, v in state_dict.items()}
    shards, cur, size = [], {}, 0
    for k, v in half.items():
        nbytes = v.numel() * v.element_size()
        if cur and size + nbytes > max_shard_mb * 1e6:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += nbytes
    shards.append(cur)
    out = []
    for i, shard in enumerate(shards, start=1):
        p = path.with_name(f"{path.stem}.part{i}of{len(shards)}{path.suffix}")
        torch.save({"mapping_net": shard, "meta": meta or {}}, p)
        out.append(p)
    return out


def load_mapping_weights(model, path: Path) -> Dict:
    """Loads (sharded) fine-tuned mapping weights into ``model.mapping_net``."""
    parts = _shard_paths(path) or [Path(path)]
    state, meta = {}, {}
    for p in parts:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state.update(ckpt["mapping_net"])
        meta = ckpt.get("meta", meta)
    ref = model.mapping_net.state_dict()
    model.mapping_net.load_state_dict({k: v.to(ref[k].dtype) for k, v in state.items()})
    model.mapping_net.eval()
    return meta


__all__ = [
    "LARGE_MISSING_THRESHOLD",
    "OpenCVBaselineRestorer",
    "setup_wan_repo",
    "load_wan_scratch_model",
    "wan_forward",
    "WanRestorer",
    "finetune_wan_mapping",
    "save_mapping_weights",
    "load_mapping_weights",
    "DEFAULT_FINETUNED_CKPT",
]
