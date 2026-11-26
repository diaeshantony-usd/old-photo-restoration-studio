"""Reference damage masks for OpenPhoto Restore (notebook 03).

OpenPhoto Restore pairs every damaged photo with its clean original but ships no damage masks.
These functions derive a mask from each pair. They are used to train the OpenPhoto-adapted damage
detector and to measure detector quality; they need the clean original, so the application never
calls them.

How a mask is built:

1. For each colour channel of the damaged photo, fit a robust quadratic tone curve that maps it to
   the clean luminance. This removes fading, sepia and contrast changes that affect the whole photo.
2. Keep only local differences (the residual minus a heavily blurred copy of itself) and take the
   largest value over the three channels. Working per channel catches scratches that differ mainly
   in colour, such as a yellow scratch on a white sky.
3. Mark two kinds of pixel as damage:
   * pixels that stand out from their own neighbourhood (thin scratches and dust in textured areas);
   * large connected regions with a clear absolute difference in smooth areas (long or soft
     scratches, tears and cracks). Strongly textured areas of the clean photo are excluded unless
     the difference is very large, because blur alone changes texture.
4. Remove specks smaller than 8 pixels and dilate by 2 pixels so inpainting has a margin.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.metrics import to_luma

_SAMPLE = np.random.default_rng(0).choice(512 * 512, 30000, replace=False)


def tone_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Robust quadratic curve mapping x to y (both H x W in 0-1), fitted on 30k sampled pixels."""
    xf, yf = x.ravel(), y.ravel()
    if xf.size == 512 * 512:
        idx = _SAMPLE
    else:
        idx = np.random.default_rng(0).choice(xf.size, min(30000, xf.size), replace=False)
    xs, ys = xf[idx], yf[idx]
    A = np.stack([xs ** 2, xs, np.ones_like(xs)], 1)
    w = np.ones_like(xs, bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[w], ys[w], rcond=None)
        r = ys - A @ coef
        w = np.abs(r) < 3 * (1.4826 * np.median(np.abs(r[w])) + 1e-6)
    return (coef[0] * xf ** 2 + coef[1] * xf + coef[2]).reshape(x.shape)


def local_residual(damaged_rgb: np.ndarray, clean_rgb: np.ndarray) -> np.ndarray:
    """Largest per-channel local difference between the tone-matched damaged photo and the clean one."""
    yc = to_luma(clean_rgb) / 255.0
    out = np.zeros(yc.shape, np.float32)
    for ch in range(3):
        x = damaged_rgb[..., ch].astype(np.float32) / 255.0
        r = (tone_fit(x, yc) - yc).astype(np.float32)
        out = np.maximum(out, np.abs(r - cv2.GaussianBlur(r, (0, 0), 8)))
    return out


def _keep_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.r_[False, stats[1:, cv2.CC_STAT_AREA] >= min_area]
    return keep[lab].astype(np.uint8)


def _local_std(y: np.ndarray, k: int = 7) -> np.ndarray:
    m = cv2.blur(y, (k, k))
    return np.sqrt(np.maximum(cv2.blur(y * y, (k, k)) - m * m, 0))


def reference_mask(damaged_rgb: np.ndarray, clean_rgb: np.ndarray, grow: int = 2) -> np.ndarray:
    """H x W uint8 damage mask (1 = damaged) derived from a damaged photo and its clean original."""
    lr = local_residual(damaged_rgb, clean_rgb)

    # (a) Pixels that stand out from their neighbourhood, grown from confident seeds.
    background = cv2.medianBlur((np.clip(lr, 0, 1) * 255).astype(np.uint8), 31).astype(np.float32) / 255.0 + 0.01
    weak = (lr > 0.04) & (lr > 3.0 * background)
    strong = weak & ((lr > 0.12) | (lr > 6.0 * background))
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    seeded = np.zeros(n, bool)
    seeded[np.unique(lab[strong])] = True
    seeded[0] = False
    a = _keep_components(seeded[lab], 8)

    # (b) Large regions with a clear absolute difference, outside strongly textured areas.
    texture = _local_std(to_luma(clean_rgb).astype(np.float32) / 255.0)
    b = _keep_components((lr > 0.07) & ((texture < 0.06) | (lr > 0.15)), 30)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    return cv2.dilate(a | b, kernel)


def mask_scores(pred: np.ndarray, ref: np.ndarray):
    """Pooled recall, precision and IoU of predicted masks against reference masks."""
    pred, ref = pred.astype(bool), ref.astype(bool)
    tp = np.logical_and(pred, ref).sum()
    return tp / max(ref.sum(), 1), tp / max(pred.sum(), 1), tp / max(np.logical_or(pred, ref).sum(), 1)


__all__ = ["tone_fit", "local_residual", "reference_mask", "mask_scores"]
