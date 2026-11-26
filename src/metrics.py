"""Image-quality metrics shared by the repair notebook (03) and the pipeline evaluation.

All functions take H x W x 3 uint8 RGB arrays.

Why luminance: OpenPhoto Restore's damage pipeline turns every photo grayscale (and
sometimes sepia), while the reference is in colour. Colour is the colorization stage's
job (DDColor, notebook 04), so the repair stage is scored on luminance (ITU-R BT.601 Y). Scoring
repair in RGB would mostly measure the missing colour, which no repair method adds.

* ``psnr_y`` / ``ssim_y``: full-image PSNR and SSIM on luminance.
* ``masked_psnr_y``: PSNR on luminance inside the damage mask only. This is the
  like-for-like comparison between methods that only touch masked pixels and methods
  that change every pixel (NLM + CLAHE, the Wan et al. network).
* ``LPIPSMetric``: real LPIPS (Zhang et al., 2018, AlexNet backbone, ``lpips`` package),
  computed on the luminance image repeated to 3 channels. It is not used as a training
  loss anywhere in this project.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity

PSNR_CAP = 100.0  # dB, returned for identical images so means stay finite


def to_luma(image_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 -> float64 luminance Y in [0, 255] (ITU-R BT.601)."""
    img = image_u8.astype(np.float64)
    if img.ndim == 2:
        return img
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def luma_rgb_u8(image_u8: np.ndarray) -> np.ndarray:
    """Grayscale version of an RGB image, still H x W x 3 uint8 (used as the repair target)."""
    y = np.clip(np.round(to_luma(image_u8)), 0, 255).astype(np.uint8)
    return np.repeat(y[..., None], 3, axis=2)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return PSNR_CAP
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def psnr_y(pred_u8: np.ndarray, target_u8: np.ndarray) -> float:
    return _psnr(to_luma(pred_u8), to_luma(target_u8))


def psnr_rgb(pred_u8: np.ndarray, target_u8: np.ndarray) -> float:
    return _psnr(pred_u8.astype(np.float64), target_u8.astype(np.float64))


def masked_psnr_y(pred_u8: np.ndarray, target_u8: np.ndarray, mask: np.ndarray) -> float:
    """Luminance PSNR over pixels where ``mask > 0``. Returns NaN for an empty mask."""
    m = mask > 0
    if not m.any():
        return float("nan")
    return _psnr(to_luma(pred_u8)[m], to_luma(target_u8)[m])


def ssim_y(pred_u8: np.ndarray, target_u8: np.ndarray) -> float:
    return float(
        structural_similarity(
            to_luma(pred_u8), to_luma(target_u8), data_range=255.0, gaussian_weights=True,
            sigma=1.5, use_sample_covariance=False,
        )
    )


class LPIPSMetric:
    """Real LPIPS distance (lower is better) on luminance images."""

    def __init__(self, device: Optional[torch.device] = None, net: str = "alex"):
        import lpips  # pip install lpips

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()

    def _prep(self, image_u8: np.ndarray) -> torch.Tensor:
        y = to_luma(image_u8).astype(np.float32) / 127.5 - 1.0  # LPIPS expects [-1, 1]
        t = torch.from_numpy(y)[None, None].repeat(1, 3, 1, 1)
        return t.to(self.device)

    @torch.no_grad()
    def __call__(self, pred_u8: np.ndarray, target_u8: np.ndarray) -> float:
        return float(self.model(self._prep(pred_u8), self._prep(target_u8)).item())


__all__ = [
    "PSNR_CAP",
    "to_luma",
    "luma_rgb_u8",
    "psnr_y",
    "psnr_rgb",
    "masked_psnr_y",
    "ssim_y",
    "LPIPSMetric",
]
