"""Inpainting and the full repair pipeline (notebook 03).

* ``LamaInpainter``: LaMa (Suvorov et al., 2022, *Resolution-robust Large Mask Inpainting with
  Fourier Convolutions*), the pretrained ``big-lama`` model. Its fast Fourier convolutions see the
  whole image, so it fills thin scratches and large tears from context anywhere in the photo. The
  TorchScript export published with the ``simple-lama-inpainting`` project is downloaded once at
  run time and called directly (that package's dependency pins do not install on current Python).
* ``TeleaInpainter``: OpenCV fast-marching inpainting, the classical reference.
* ``RepairPipeline``: OpenPhoto-adapted damage detector -> learned tone correction -> inpainting of
  the detected damage. Only masked pixels are inpainted; every other pixel is the tone-corrected
  photo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LARGE_MISSING_THRESHOLD = 0.20  # more than 20% of pixels masked => repair flagged as low confidence


class LamaInpainter:
    """Pretrained LaMa. ``__call__(image_rgb, mask)`` returns the image with masked pixels filled."""

    def __init__(self, cache_path: Path, device="cuda"):
        cache_path = Path(cache_path)
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(LAMA_URL, str(cache_path))
        self.device = torch.device(device)
        self.model = torch.jit.load(str(cache_path), map_location=self.device).eval()

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return image_rgb
        h, w = mask.shape
        x = torch.from_numpy(image_rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
        m = torch.from_numpy((mask > 0).astype(np.float32))[None, None].to(self.device)
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8          # LaMa needs sizes divisible by 8
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        m = F.pad(m, (0, pw, 0, ph))
        out = self.model(x, m)[0, :, :h, :w].clamp(0, 1)
        filled = (out.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        result = image_rgb.copy()
        result[mask > 0] = filled[mask > 0]                  # only masked pixels change
        return result


class TeleaInpainter:
    """OpenCV Telea fast-marching inpainting (classical reference)."""

    def __init__(self, radius: int = 5):
        self.radius = radius

    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return image_rgb
        return cv2.inpaint(image_rgb, (mask > 0).astype(np.uint8) * 255, self.radius, cv2.INPAINT_TELEA)


class RepairPipeline:
    """Shipped repair stage: detect damage, correct tone, inpaint the detected damage.

    ``restore(damaged_rgb)`` runs the detector itself; ``restore(damaged_rgb, mask)`` uses a given mask.
    After each call ``last_info`` holds the mask, its coverage and a low-confidence flag.
    """

    def __init__(self, detector, tone, inpainter):
        self.detector, self.tone, self.inpainter = detector, tone, inpainter
        self.last_info: dict = {}

    def restore(self, damaged_rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is None:
            mask = self.detector.predict(damaged_rgb)
        ratio = float((mask > 0).mean())
        self.last_info = {"mask": mask, "mask_ratio": ratio, "low_confidence": ratio > LARGE_MISSING_THRESHOLD,
                          "note": "large missing region: result is a plausible fill, not a recovery" if ratio > LARGE_MISSING_THRESHOLD else ""}
        return self.inpainter(self.tone(damaged_rgb), mask)


__all__ = ["LAMA_URL", "LamaInpainter", "TeleaInpainter", "RepairPipeline"]
