"""OpenPhoto Restore loading for the repair notebook (03).

Dataset: joshuachin/openphoto-restore-dataset (Hugging Face, CC BY 4.0). 5,000 Unsplash
photos, each with a synthetically damaged copy (cracks, scratches, dust, grayscale/sepia,
fading, grain). The dataset ships an official ``train`` (4,500) and ``test`` (500) split
of different source photos.

Splits used by the repair notebook (all disjoint by source photo):
  * train: official ``train`` minus the validation slice  (4,050 photos)
  * val:   a fixed, seeded 10% slice of official ``train``  (450 photos), used only for
           choosing the fine-tuning checkpoint
  * test:  official ``test``                                (500 photos), touched once,
           for the final table and the verification figure

Images are decoded once, resized so the short side is ``size`` pixels and centre-cropped
to ``size x size``; the same transform is applied to the pristine and damaged copy so the
pair stays pixel-aligned.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

HF_REPO_ID = "joshuachin/openphoto-restore-dataset"


def _square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size]


@dataclass
class OpenPhotoArrays:
    """Decoded, aligned uint8 arrays for one split."""

    pristine: np.ndarray  # (N, size, size, 3) uint8 RGB, colour reference
    damaged: np.ndarray   # (N, size, size, 3) uint8 RGB
    ids: np.ndarray       # (N,) int, row index in the official HF split
    split: str

    def __len__(self) -> int:
        return len(self.ids)


def load_openphoto_split(
    hf_split: str,
    size: int = 512,
    cache_dir: Optional[Path] = None,
) -> OpenPhotoArrays:
    """Downloads (once) and decodes one official split into aligned uint8 arrays.

    A decoded ``.npz`` cache is written next to the HF cache so reruns skip decoding.
    """
    from datasets import load_dataset  # pip install datasets

    cache_dir = Path(cache_dir) if cache_dir is not None else None
    npz_path = cache_dir / f"openphoto_{hf_split}_{size}.npz" if cache_dir else None

    if npz_path is not None and npz_path.exists():
        data = np.load(npz_path)
        pristine, damaged, ids = data["pristine"], data["damaged"], data["ids"]
    else:
        ds = load_dataset(HF_REPO_ID, split=hf_split, cache_dir=str(cache_dir) if cache_dir else None)
        n = len(ds)
        pristine = np.empty((n, size, size, 3), dtype=np.uint8)
        damaged = np.empty((n, size, size, 3), dtype=np.uint8)
        for i in range(n):
            row = ds[i]
            p = np.asarray(row["pristine_image"].convert("RGB"))
            d = np.asarray(row["damaged_image"].convert("RGB"))
            if d.shape[:2] != p.shape[:2]:  # keep the pair aligned
                d = cv2.resize(d, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA)
            pristine[i] = _square(p, size)
            damaged[i] = _square(d, size)
        ids = np.arange(n)
        if npz_path is not None:
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz_path, pristine=pristine, damaged=damaged, ids=ids)

    return OpenPhotoArrays(pristine=pristine, damaged=damaged, ids=ids, split=hf_split)


def subset(arrays: OpenPhotoArrays, rows: np.ndarray) -> OpenPhotoArrays:
    """Rows of an already-decoded split (no re-decoding)."""
    return OpenPhotoArrays(
        pristine=arrays.pristine[rows], damaged=arrays.damaged[rows], ids=arrays.ids[rows], split=arrays.split
    )


def train_val_rows(n_train_total: int = 4500, val_fraction: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Seeded split of the official train rows into (train_rows, val_rows), disjoint."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train_total)
    n_val = int(round(n_train_total * val_fraction))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def check_disjoint(*splits: OpenPhotoArrays) -> None:
    """Asserts no source photo appears in two splits (ids are per HF split)."""
    seen: Dict[Tuple[str, int], str] = {}
    for s in splits:
        for i in s.ids.tolist():
            key = (s.split, int(i))
            assert key not in seen, f"photo {key} is in both {seen[key]} and {s}"
            seen[key] = repr(s)


class OpenPhotoRepairDataset(Dataset):
    """Random (train) or centre (eval) crops of aligned damaged / grayscale target / mask.

    The target is the *grayscale* pristine photo (see src/metrics.py for why). Masks are
    the damage detector's predictions on the damaged photo, computed once per image and
    passed in, so training sees the same kind of mask the repair stage gets at inference.
    """

    def __init__(
        self,
        arrays: OpenPhotoArrays,
        masks: np.ndarray,
        crop_size: int = 256,
        is_training: bool = True,
    ):
        assert masks.shape[0] == len(arrays)
        self.arrays = arrays
        self.masks = masks
        self.crop = crop_size
        self.is_training = is_training
        from src.metrics import luma_rgb_u8

        self.targets = np.stack([luma_rgb_u8(p) for p in arrays.pristine])

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dmg, tgt, msk = self.arrays.damaged[idx], self.targets[idx], self.masks[idx]
        h, w = dmg.shape[:2]
        c = self.crop
        if self.is_training:
            y, x = random.randint(0, h - c), random.randint(0, w - c)
        else:
            y, x = (h - c) // 2, (w - c) // 2
        dmg, tgt, msk = dmg[y : y + c, x : x + c], tgt[y : y + c, x : x + c], msk[y : y + c, x : x + c]
        if self.is_training and random.random() < 0.5:
            dmg, tgt, msk = dmg[:, ::-1], tgt[:, ::-1], msk[:, ::-1]

        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
        return {
            "damaged": to_t(dmg),
            "clean": to_t(tgt),
            "mask": torch.from_numpy(np.ascontiguousarray(msk)).float()[None],
            "id": int(self.arrays.ids[idx]),
        }


__all__ = [
    "HF_REPO_ID",
    "OpenPhotoArrays",
    "load_openphoto_split",
    "subset",
    "train_val_rows",
    "check_disjoint",
    "OpenPhotoRepairDataset",
]
