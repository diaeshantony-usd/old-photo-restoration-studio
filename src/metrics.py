"""Evaluation metrics shared by every notebook, the tests and the demo.

Images are ``H x W x 3`` uint8 RGB numpy arrays unless stated otherwise.
Heavy dependencies (scikit-image, lpips) are imported inside the functions that
need them, so importing this module is cheap.

Metric   Measures                                    Better  Used for
-------  ------------------------------------------  ------  --------------------------
PSNR     pixel-level closeness (dB)                  higher  repair, enhancement, pipeline
SSIM     edge / texture structure, 0-1               higher  repair, enhancement, pipeline
LPIPS    perceptual distance from deep features      lower   repair, colorization, pipeline
IoU, F1  overlap of predicted and true damage masks  higher  damage detection
Delta E  CIEDE2000 colour difference in Lab          lower   colorization
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_LPIPS_MODELS: dict = {}


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"image shapes differ: {a.shape} vs {b.shape}")


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB. Identical images return ``inf``."""
    _check_pair(pred, target)
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 255.0) -> float:
    """Structural similarity (Wang et al., 2004), averaged over colour channels."""
    from skimage.metrics import structural_similarity

    _check_pair(pred, target)
    channel_axis = 2 if pred.ndim == 3 else None
    return float(structural_similarity(target, pred, channel_axis=channel_axis, data_range=data_range))


def lpips_distance(pred: np.ndarray, target: np.ndarray, device: Optional[str] = None, net: str = "alex") -> float:
    """Learned perceptual image patch similarity (Zhang et al., 2018). Lower is better."""
    import lpips
    import torch

    _check_pair(pred, target)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (net, device)
    if key not in _LPIPS_MODELS:
        _LPIPS_MODELS[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    model = _LPIPS_MODELS[key]

    def to_tensor(x: np.ndarray):
        return (torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        return float(model(to_tensor(pred), to_tensor(target)).item())


def delta_e(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean CIEDE2000 colour difference in Lab space (about 1 is barely visible)."""
    from skimage.color import deltaE_ciede2000, rgb2lab

    _check_pair(pred, target)
    return float(deltaE_ciede2000(rgb2lab(target), rgb2lab(pred)).mean())


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler and Susstrunk (2003) colourfulness. Near 0 for gray, higher for vivid images."""
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def iou_f1(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    """IoU and F1 (Dice) of two binary damage masks. Two empty masks score 1.0."""
    _check_pair(pred_mask, true_mask)
    p, t = pred_mask.astype(bool), true_mask.astype(bool)
    inter = float(np.logical_and(p, t).sum())
    union = float(np.logical_or(p, t).sum())
    total = float(p.sum() + t.sum())
    if union == 0:
        return {"iou": 1.0, "f1": 1.0}
    return {"iou": (inter + eps) / (union + eps), "f1": (2 * inter + eps) / (total + eps)}


def image_metrics(pred: np.ndarray, target: np.ndarray, device: Optional[str] = None) -> Dict[str, float]:
    """PSNR, SSIM and LPIPS in one call (repair, enhancement, whole pipeline)."""
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "lpips": lpips_distance(pred, target, device=device),
    }
