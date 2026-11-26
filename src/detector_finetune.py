"""OpenPhoto-adapted damage detector (notebook 03).

Starts from the U-Net trained in notebook 02 (``models/damage_detection_unet.pt``, unchanged) and
fine-tunes a copy on OpenPhoto Restore, using the reference masks from ``src/reference_masks.py``
as labels. The adapted weights are saved under their own name,
``models/damage_detection_unet_openphoto.pt``, and are loaded with the standard
``DamageDetector`` class.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.damage_detection import DamageDetectionUNet, predict_damage_mask_tiled
from src.reference_masks import mask_scores


class MaskCrops(torch.utils.data.Dataset):
    """Random square crops of damaged photos with their masks; horizontal and vertical flips."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, size: int = 256, crops_per_epoch: int = 8000):
        self.images, self.masks, self.size, self.n = images, masks, size, crops_per_epoch

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _):
        i = np.random.randint(len(self.images))
        h, w = self.masks[i].shape
        y, x = np.random.randint(0, h - self.size + 1), np.random.randint(0, w - self.size + 1)
        im = self.images[i][y:y + self.size, x:x + self.size].astype(np.float32) / 255.0
        m = self.masks[i][y:y + self.size, x:x + self.size].astype(np.float32)
        if np.random.rand() < 0.5:
            im, m = im[:, ::-1], m[:, ::-1]
        if np.random.rand() < 0.5:
            im, m = im[::-1], m[::-1]
        return torch.from_numpy(im.copy()).permute(2, 0, 1), torch.from_numpy(m.copy())[None]


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    union = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def predict_masks(model: torch.nn.Module, images, device, threshold: float = 0.5) -> np.ndarray:
    """Binary masks for a stack of RGB uint8 photos (same post-processing as DamageDetector)."""
    return np.stack([predict_damage_mask_tiled(model, im, device, tile_size=512, overlap=64, threshold=threshold,
                                               dilate_radius=1)[0] for im in images]).astype(np.uint8)


def finetune_detector(base_checkpoint: Path, train_images, train_masks, val_images, val_masks, device,
                      epochs: int = 12, batch_size: int = 16, lr: float = 1e-3, seed: int = 0,
                      log: Optional[Callable[[str], None]] = print):
    """Fine-tunes the notebook-02 U-Net on (image, mask) pairs; keeps the epoch with the best validation IoU.

    Loss: weighted binary cross-entropy (damage pixels are rare) plus Dice loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DamageDetectionUNet().to(device)
    model.load_state_dict(torch.load(base_checkpoint, map_location=device, weights_only=True))
    pos = float(train_masks.mean())
    pos_weight = torch.tensor([min(10.0, (1 - pos) / max(pos, 1e-6))], device=device)
    loader = torch.utils.data.DataLoader(MaskCrops(train_images, train_masks), batch_size=batch_size, shuffle=True,
                                         num_workers=4, drop_last=True, generator=torch.Generator().manual_seed(seed),
                                         worker_init_fn=lambda w: np.random.seed(seed * 100 + w))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    best, history, t0 = (-1.0, None, 0), [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        recall, precision, iou = mask_scores(predict_masks(model, val_images, device), val_masks)
        history.append({"epoch": ep, "loss": total / len(loader), "recall": recall, "precision": precision, "iou": iou})
        if log:
            log(f"Epoch {ep:2d}: loss {total / len(loader):.3f} | validation recall {recall:.3f} "
                f"precision {precision:.3f} IoU {iou:.3f} | {time.time() - t0:.0f} s")
        if iou > best[0]:
            best = (iou, copy.deepcopy(model.state_dict()), ep)
    model.load_state_dict(best[1])
    return model.eval(), {"best_val_iou": best[0], "best_epoch": best[2], "history": history}


__all__ = ["MaskCrops", "dice_loss", "predict_masks", "finetune_detector"]
