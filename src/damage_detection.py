"""Damage detection (notebook 02): U-Net detector and classical OpenCV detector.

Every detector exposes the same interface, used by the repair notebook (03):

    class DamageDetector:
        def predict(self, image: np.ndarray) -> np.ndarray:  # H x W uint8 mask, 1 = damaged
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "damage_detection_unet.pt"


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsamples spatial resolution by 2x via MaxPool, then applies ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsamples features by 2x, concatenates skip connection, then applies ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
        return self.conv(torch.cat([skip, x], dim=1))


class DamageDetectionUNet(nn.Module):
    """Lightweight 4-level U-Net for physical photo defect segmentation (~1.95M params)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.inc = ConvBlock(in_channels, 32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)

        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up3 = UpBlock(64, 32, 32)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def predict_damage_mask_tiled(
    model: nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
    threshold: float = 0.5,
    dilate_radius: int = 1,
    min_blob_area: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predicts a full-resolution binary damage mask using 2D Hann-window tile blending.

    Returns:
        binary_mask: (H, W) uint8 array in {0, 1} (1 = damaged).
        prob_map: (H, W) float32 probability map in [0.0, 1.0].
    """
    model.eval()
    if image_np.dtype == np.uint8:
        img_f = image_np.astype(np.float32) / 255.0
    else:
        img_f = np.clip(image_np.astype(np.float32), 0.0, 1.0)

    h, w = img_f.shape[:2]
    stride = max(1, tile_size - overlap)

    canvas = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    win_y = np.hanning(tile_size + 2)[1:-1]
    win_x = np.hanning(tile_size + 2)[1:-1]
    window = np.outer(win_y, win_x).astype(np.float32)

    y_points = list(range(0, max(1, h - tile_size + 1), stride))
    if h > tile_size and y_points[-1] + tile_size < h:
        y_points.append(h - tile_size)

    x_points = list(range(0, max(1, w - tile_size + 1), stride))
    if w > tile_size and x_points[-1] + tile_size < w:
        x_points.append(w - tile_size)

    with torch.no_grad():
        for y in y_points:
            for x in x_points:
                patch = img_f[y : y + tile_size, x : x + tile_size]
                ph, pw = patch.shape[:2]
                if ph < tile_size or pw < tile_size:
                    patch = np.pad(
                        patch,
                        ((0, tile_size - ph), (0, tile_size - pw), (0, 0)),
                        mode="reflect",
                    )

                tensor = (
                    torch.from_numpy(patch.astype(np.float32))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

                canvas[y : y + ph, x : x + pw] += prob[:ph, :pw] * window[:ph, :pw]
                weights[y : y + ph, x : x + pw] += window[:ph, :pw]

    prob_map = np.clip(canvas / np.maximum(weights, 1e-6), 0.0, 1.0)
    raw_mask = (prob_map > threshold).astype(np.uint8)

    if min_blob_area > 1 and raw_mask.sum() > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw_mask, connectivity=8
        )
        filtered = np.zeros_like(raw_mask)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_blob_area:
                filtered[labels == idx] = 1
        raw_mask = filtered

    if dilate_radius > 0 and raw_mask.sum() > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1)
        )
        raw_mask = cv2.dilate(raw_mask, k, iterations=1)

    return raw_mask.astype(np.uint8), prob_map


class DamageDetector:
    """Trained U-Net damage detector (notebook 02)."""

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
        threshold: float = 0.5,
        dilate_radius: int = 1,
        tile_size: int = 512,
        overlap: int = 64,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold
        self.dilate_radius = dilate_radius
        self.tile_size = tile_size
        self.overlap = overlap

        self.model = DamageDetectionUNet(in_channels=3, out_channels=1).to(self.device)
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.eval()
        self.last_prob_map: Optional[np.ndarray] = None

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predicts an H x W uint8 binary damage mask (1 = damaged, 0 = clean)."""
        mask, prob_map = predict_damage_mask_tiled(
            self.model,
            image,
            self.device,
            tile_size=self.tile_size,
            overlap=self.overlap,
            threshold=self.threshold,
            dilate_radius=self.dilate_radius,
        )
        self.last_prob_map = prob_map
        return mask


class OpenCVDamageDetector:
    """Classical morphological threshold detector, used as the OpenCV baseline in notebook 02."""

    def __init__(self, threshold: int = 38, dilate_radius: int = 1):
        self.threshold = threshold
        self.dilate_radius = dilate_radius

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Detects bright/dark scratches and dust specks via top-hat and black-hat morphology."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        combined = cv2.max(tophat, blackhat)
        mask = (combined > self.threshold).astype(np.uint8)

        if self.dilate_radius > 0 and mask.sum() > 0:
            dk = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.dilate_radius + 1, 2 * self.dilate_radius + 1),
            )
            mask = cv2.dilate(mask, dk, iterations=1)
        return mask
