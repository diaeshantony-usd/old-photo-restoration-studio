"""Colorization stage: DDColor (Kang et al., 2023) plus the "already in color" skip rule.

Develop and explain this stage in ``notebooks/04_colorization``; the code lives here so the
notebook, the tests and the Gradio demo all run the same implementation.

DDColor itself is imported lazily from a clone of https://github.com/piddnad/DDColor, so this
module can be imported (and the skip rule tested) without the model installed.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

log = logging.getLogger("colorizer")

DEFAULT_MODEL_ID = "piddnad/ddcolor_modelscope"   # memory fallback: "piddnad/ddcolor_paper_tiny"
DEFAULT_INPUT_SIZE = 512

# Skip-rule thresholds (Lab chroma / hue spread). First guesses; tuned in notebook 04.
GRAY_CHROMA = 6.0       # mean Lab chroma below this -> gray
SEPIA_CHROMA = 30.0     # mean chroma below this AND one dominant hue -> sepia
SEPIA_HUE_STD = 0.35    # radians; sepia pixels all share one hue

_SEPIA_MATRIX = np.array([[0.393, 0.769, 0.189],
                          [0.349, 0.686, 0.168],
                          [0.272, 0.534, 0.131]])


# --------------------------------------------------------------------------- skip rule
def chroma_stats(rgb: np.ndarray) -> Tuple[float, float]:
    """Return (mean Lab chroma, circular std of hue in radians) for a uint8 RGB image."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    mean_chroma = float(chroma.mean())
    mask = chroma > 2.0                      # ignore near-neutral pixels when measuring hue spread
    if mask.sum() < 50:
        return mean_chroma, 0.0
    hue = np.arctan2(b[mask], a[mask])
    r = np.hypot(np.cos(hue).mean(), np.sin(hue).mean())    # 1.0 = all pixels share one hue
    return mean_chroma, float(np.sqrt(-2.0 * np.log(max(r, 1e-6))))


def classify_color(rgb: np.ndarray) -> str:
    """Return 'gray', 'sepia' or 'color'."""
    chroma, hue_std = chroma_stats(rgb)
    if chroma < GRAY_CHROMA:
        return "gray"
    if chroma < SEPIA_CHROMA and hue_std < SEPIA_HUE_STD:
        return "sepia"
    return "color"


def is_grayscale(rgb: np.ndarray) -> bool:
    """True when the photo should be colorized (gray or sepia)."""
    return classify_color(rgb) in ("gray", "sepia")


def to_gray(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.stack([g] * 3, axis=-1)


def to_sepia(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) @ _SEPIA_MATRIX.T, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- model
def load_ddcolor(model_id: str = DEFAULT_MODEL_ID, input_size: int = DEFAULT_INPUT_SIZE,
                 ddcolor_dir: Optional[str] = None):
    """Load a pretrained DDColor pipeline from Hugging Face.

    ``ddcolor_dir`` is a clone of the DDColor repository; it is added to ``sys.path``.
    """
    if ddcolor_dir and ddcolor_dir not in sys.path:
        sys.path.insert(0, ddcolor_dir)
    from inference.colorization_pipeline_hf import DDColorHF, ImageColorizationPipelineHF

    model = DDColorHF.from_pretrained(model_id)
    return ImageColorizationPipelineHF(model=model, input_size=input_size)


class Colorizer:
    """DDColor wrapper with the 'already in color' skip rule.

    ``last_info`` records what happened on the latest call: ``ran``, ``kind``, ``reason``.
    """

    def __init__(self, pipeline=None, enabled: bool = True, **load_kwargs):
        self.pipeline = pipeline if pipeline is not None else load_ddcolor(**load_kwargs)
        self.enabled = enabled
        self.last_info: dict = {}

    @torch.inference_mode()
    def colorize(self, image: np.ndarray, force: bool = False) -> np.ndarray:
        """image: H x W x 3 uint8 RGB. Returns uint8 RGB of the same size."""
        kind = classify_color(image)
        if not self.enabled:
            self.last_info = dict(ran=False, kind=kind, reason="colorization switched off")
        elif kind == "color" and not force:
            self.last_info = dict(ran=False, kind=kind, reason="photo already has color")
        else:
            self.last_info = dict(ran=True, kind=kind, reason="gray/sepia input colorized")
        log.info("%s", self.last_info)
        if not self.last_info["ran"]:
            return image

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        use_cuda = torch.cuda.is_available()
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                out_bgr = self.pipeline.process(bgr)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning("out of GPU memory: retrying at input_size=256")
            old = self.pipeline.input_size
            self.pipeline.input_size = 256
            try:
                out_bgr = self.pipeline.process(bgr)
            finally:
                self.pipeline.input_size = old
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        if out.shape[:2] != image.shape[:2]:
            out = cv2.resize(out, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        return out
