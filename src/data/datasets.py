# ==============================================================================
# Old Photo Restoration Studio - Centralized Dataset Architecture
# University of San Diego - Applied Computer Vision for AI (AAI-521)
# ==============================================================================
"""
Centralized PyTorch Dataset abstractions and data collection utilities
for all stages of the Old Photo Restoration Studio pipeline:
  - Stage 1: Ingestion & Preprocessing Verification
  - Stage 2: Physical Damage & Scratch Detection
  - Repair (notebook 03): Defect Inpainting & Latent Repair
  - Stage 4: Face Enhancement & Archival Restoration
  - Stage 5: Historical Colorization & Degradation Removal
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Lift decompression bomb ceiling for archival high-resolution film scans
Image.MAX_IMAGE_PIXELS = None


# ==============================================================================
# 1. Damage Detection Dataset (Stages 1 & 2)
# ==============================================================================

class DamageDataset(Dataset):
    """
    Standardized PyTorch Dataset loading paired degraded photographs and binary damage masks.
    
    Accepts samples from real archival scans (ARTeFACT) or synthetic procedural defects.
    Applies synchronized geometric data augmentations during training and deterministic
    center crops during validation. Implements crop-first decompression to maintain a
    constant memory profile on gigapixel historical scans.
    """
    def __init__(
        self,
        samples: Union[List[Tuple[Path, Path]], Path, Tuple[Path, Path]],
        crop_size: int = 256,
        is_training: bool = True
    ):
        """
        Args:
            samples: List of (image_path, mask_path) tuples, or a tuple of (images_dir, masks_dir).
            crop_size: Spatial crop dimension for model input patches (default 256).
            is_training: If True, applies random crops and synchronized flips. If False, center crop.
        """
        if isinstance(samples, (tuple, list)) and len(samples) == 2 and isinstance(samples[0], Path) and samples[0].is_dir():
            images_dir, masks_dir = samples[0], samples[1]
            self.samples = [
                (p, masks_dir / p.name)
                for p in sorted(images_dir.glob("*.png"))
                if (masks_dir / p.name).exists()
            ]
        elif isinstance(samples, list):
            self.samples = samples
        else:
            raise ValueError(f"Invalid samples argument provided: {samples}")

        self.crop_size = crop_size
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_path, mask_path = self.samples[idx]

        # Crop-first decompression to prevent memory saturation on large archival scans
        with Image.open(img_path) as img_raw, Image.open(mask_path) as mask_raw:
            w, h = img_raw.size
            if w < self.crop_size or h < self.crop_size:
                scale = max(self.crop_size / w, self.crop_size / h)
                new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
                img_raw = img_raw.resize((new_w, new_h), Image.Resampling.BILINEAR)
                mask_raw = mask_raw.resize((new_w, new_h), Image.Resampling.NEAREST)
                w, h = new_w, new_h

            if self.is_training:
                x = random.randint(0, w - self.crop_size)
                y = random.randint(0, h - self.crop_size)
            else:
                x = (w - self.crop_size) // 2
                y = (h - self.crop_size) // 2

            img_crop = img_raw.crop((x, y, x + self.crop_size, y + self.crop_size)).convert("RGB")
            mask_crop = mask_raw.crop((x, y, x + self.crop_size, y + self.crop_size)).convert("L")

        # Synchronized geometric augmentations during training
        if self.is_training:
            if random.random() < 0.5:
                img_crop = img_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask_crop = mask_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img_crop = img_crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask_crop = mask_crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        img_arr = np.array(img_crop, dtype=np.float32) / 255.0
        mask_arr = np.array(mask_crop, dtype=np.float32) / 255.0

        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # (3, H, W)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)   # (1, H, W)
        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "sample_id": img_path.stem
        }


# ==============================================================================
# 2. Synthetic Triplet Dataset (Stages 1 & 3)
# ==============================================================================

class SyntheticTripletDataset(Dataset):
    """
    High-throughput PyTorch Dataset streaming pre-cached synthetic triplets from disk.
    
    Loads aligned damaged, clean, and mask files simultaneously, feeding downstream
    damage detection (Stage 2) and inpainting repair (notebook 03) networks.
    """
    def __init__(
        self,
        split_dir: Path,
        crop_size: Optional[int] = None,
        is_training: bool = True
    ):
        """
        Args:
            split_dir: Root directory for split containing damaged/, clean/, masks/.
            crop_size: Optional spatial dimension to crop (None keeps original resolution).
            is_training: Whether to apply random crops (True) or center crops (False).
        """
        self.dmg_dir = split_dir / "damaged"
        self.clean_dir = split_dir / "clean"
        self.mask_dir = split_dir / "masks"
        self.crop_size = crop_size
        self.is_training = is_training

        self.filenames = sorted([
            f.name for f in self.dmg_dir.glob("*.png")
            if (self.clean_dir / f.name).exists() and (self.mask_dir / f.name).exists()
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        fname = self.filenames[idx]
        dmg_path = self.dmg_dir / fname
        clean_path = self.clean_dir / fname
        mask_path = self.mask_dir / fname

        with Image.open(dmg_path) as dmg_raw, Image.open(clean_path) as clean_raw, Image.open(mask_path) as mask_raw:
            if self.crop_size is not None:
                w, h = dmg_raw.size
                if w < self.crop_size or h < self.crop_size:
                    scale = max(self.crop_size / w, self.crop_size / h)
                    nw, nh = int(w * scale) + 1, int(h * scale) + 1
                    dmg_raw = dmg_raw.resize((nw, nh), Image.Resampling.BILINEAR)
                    clean_raw = clean_raw.resize((nw, nh), Image.Resampling.BILINEAR)
                    mask_raw = mask_raw.resize((nw, nh), Image.Resampling.NEAREST)
                    w, h = nw, nh

                if self.is_training:
                    x = random.randint(0, w - self.crop_size)
                    y = random.randint(0, h - self.crop_size)
                else:
                    x = (w - self.crop_size) // 2
                    y = (h - self.crop_size) // 2

                dmg_crop = dmg_raw.crop((x, y, x + self.crop_size, y + self.crop_size)).convert("RGB")
                clean_crop = clean_raw.crop((x, y, x + self.crop_size, y + self.crop_size)).convert("RGB")
                mask_crop = mask_raw.crop((x, y, x + self.crop_size, y + self.crop_size)).convert("L")
            else:
                dmg_crop = dmg_raw.convert("RGB")
                clean_crop = clean_raw.convert("RGB")
                mask_crop = mask_raw.convert("L")

        dmg_t = torch.from_numpy(np.array(dmg_crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
        clean_t = torch.from_numpy(np.array(clean_crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.array(mask_crop, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_t = (mask_t > 0.5).float()

        return {
            "damaged": dmg_t,
            "clean": clean_t,
            "mask": mask_t,
            "sample_id": fname
        }


# ==============================================================================
# 3. Face Restoration Dataset (Stage 4)
# ==============================================================================

class FaceRestorationDataset(Dataset):
    """
    PyTorch Dataset loading high-resolution historical portrait imagery (humanface8000).
    
    Streams paired degraded/grayscale portraits and clean ground-truth references
    for facial enhancement, super-resolution, and perceptual restoration.
    """
    def __init__(
        self,
        base_dir: Path,
        crop_size: int = 512,
        is_training: bool = True
    ):
        """
        Args:
            base_dir: Directory containing color/ and gray/ subdirectories.
            crop_size: Spatial dimension for face patches (default 512).
            is_training: Whether to apply random crops (True) or center crops (False).
        """
        self.color_dir = base_dir / "color"
        self.gray_dir = base_dir / "gray"
        self.crop_size = crop_size
        self.is_training = is_training

        self.filenames = sorted([
            f.name for f in self.color_dir.glob("*.png")
            if (self.gray_dir / f.name).exists()
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        fname = self.filenames[idx]
        color_path = self.color_dir / fname
        gray_path = self.gray_dir / fname

        with Image.open(color_path) as col_raw, Image.open(gray_path) as gray_raw:
            w, h = col_raw.size
            if w < self.crop_size or h < self.crop_size:
                scale = max(self.crop_size / w, self.crop_size / h)
                nw, nh = int(w * scale) + 1, int(h * scale) + 1
                col_raw = col_raw.resize((nw, nh), Image.Resampling.BILINEAR)
                gray_raw = gray_raw.resize((nw, nh), Image.Resampling.BILINEAR)
                w, h = nw, nh

            if self.is_training:
                x = random.randint(0, w - self.crop_size)
                y = random.randint(0, h - self.crop_size)
            else:
                x = (w - self.crop_size) // 2
                y = (h - self.crop_size) // 2

            col_crop = col_raw.crop((x, y, x + self.crop_size, y + self.crop_size)).convert("RGB")
            gray_crop = gray_raw.crop((x, y, x + self.crop_size, y + self.crop_size)).convert("RGB")

        col_t = torch.from_numpy(np.array(col_crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
        gray_t = torch.from_numpy(np.array(gray_crop, dtype=np.float32) / 255.0).permute(2, 0, 1)

        return {
            "degraded": gray_t,
            "reference": col_t,
            "sample_id": fname
        }


# ==============================================================================
# 4. Colorization & Degradation Dataset (Stage 5)
# ==============================================================================

class ColorizationDataset(Dataset):
    """
    PyTorch Dataset loading historical photographs for deep colorization training.
    
    Extracts single-channel luminance (grayscale) representations as network inputs
    and full 3-channel sRGB color photographs as ground-truth targets.
    """
    def __init__(
        self,
        images_dir: Path,
        crop_size: int = 256,
        is_training: bool = True
    ):
        """
        Args:
            images_dir: Directory containing standardized clean color photographs.
            crop_size: Spatial dimension for training patches (default 256).
            is_training: Whether to apply random crops and horizontal flips.
        """
        self.images_dir = images_dir
        self.crop_size = crop_size
        self.is_training = is_training
        self.filepaths = sorted(list(self.images_dir.glob("*.png")))

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        path = self.filepaths[idx]

        with Image.open(path) as img_raw:
            w, h = img_raw.size
            if w < self.crop_size or h < self.crop_size:
                scale = max(self.crop_size / w, self.crop_size / h)
                nw, nh = int(w * scale) + 1, int(h * scale) + 1
                img_raw = img_raw.resize((nw, nh), Image.Resampling.BILINEAR)
                w, h = nw, nh

            if self.is_training:
                x = random.randint(0, w - self.crop_size)
                y = random.randint(0, h - self.crop_size)
            else:
                x = (w - self.crop_size) // 2
                y = (h - self.crop_size) // 2

            crop = img_raw.crop((x, y, x + self.crop_size, y + self.crop_size)).convert("RGB")

        if self.is_training and random.random() < 0.5:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        col_arr = np.array(crop, dtype=np.float32) / 255.0
        # Compute ITU-R BT.601 perceptual luminance channel
        gray_arr = 0.299 * col_arr[:, :, 0] + 0.587 * col_arr[:, :, 1] + 0.114 * col_arr[:, :, 2]

        color_t = torch.from_numpy(col_arr).permute(2, 0, 1)  # (3, H, W)
        gray_t = torch.from_numpy(gray_arr).unsqueeze(0)      # (1, H, W)

        return {
            "grayscale": gray_t,
            "color": color_t,
            "sample_id": path.stem
        }


# ==============================================================================
# 5. Dataset Discovery & Partitioning Utilities
# ==============================================================================

def collect_damage_pairs(
    cleaned_dir: Path,
    synthetic_dir: Path
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Discovers paired image and defect mask file paths for real ARTeFACT and synthetic data.
    
    Returns:
        artefact_pairs: List of (image_path, mask_path) for real archival scans.
        synthetic_pairs: List of (damaged_path, mask_path) for synthetic procedural damage.
    """
    # 1. ARTeFACT real historical pairs
    art_img_dir = cleaned_dir / "artefact" / "images"
    art_mask_dir = cleaned_dir / "artefact" / "masks"
    artefact_pairs = [
        (p, art_mask_dir / p.name)
        for p in sorted(art_img_dir.glob("*.png"))
        if (art_mask_dir / p.name).exists()
    ]

    # 2. Synthetic damage pairs
    synth_dmg_dir = synthetic_dir / "train" / "damaged"
    synth_mask_dir = synthetic_dir / "train" / "masks"
    synthetic_pairs = [
        (p, synth_mask_dir / p.name)
        for p in sorted(synth_dmg_dir.glob("*.png"))
        if (synth_mask_dir / p.name).exists()
    ]

    return artefact_pairs, synthetic_pairs
