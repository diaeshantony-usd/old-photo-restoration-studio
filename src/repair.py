"""Stage 3 Repair module: Wan et al. (2020) Dual-VAE Latent-Space Restoration and OpenCV Baseline.

Implements the three repair conditions specified in Section 3.5 of the Design Document:
  1. Classical OpenCV Baseline:
     - OpenCV Telea Fast Marching inpainting on the detected damage mask
     - Non-Local Means denoising (cv2.fastNlMeansDenoisingColored)
     - CLAHE contrast enhancement in CIE LAB space
  2. Zero-Shot Latent-Space Restoration (Wan et al., 2020):
     - Dual Variational Autoencoders (VAE_1 for damaged photos, VAE_2 for clean photos)
     - Latent Mapping Network with a Partial Non-Local Block that copies texture from
       undamaged latent regions (M = 0) into masked damaged regions (M = 1)
  3. Fine-Tuned Latent-Space Restoration (Wan et al., 2020):
     - Both VAEs frozen (Section 3.4); only the Latent Mapping Network is fine-tuned
       with L1 + VGG/Perceptual loss across 3 seeds.

Conforms to the standardized Restorer interface:
    class Restorer:
        def restore(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPAIR_CKPT = ROOT_DIR / "models" / "repair_wan_latent_vae.pt"
LARGE_MISSING_THRESHOLD = 0.20  # Section 8.1: >20% masked flags low confidence


# ==============================================================================
# 1. Classical OpenCV Baseline (Section 3.5, Condition 1)
# ==============================================================================


class OpenCVBaselineRestorer:
    """Classical image repair baseline: Telea inpainting + Non-Local Means denoising + CLAHE."""

    def __init__(
        self,
        inpaint_radius: int = 3,
        denoise_h: float = 5.0,
        clahe_clip: float = 1.8,
        apply_denoise_and_clahe: bool = True,
    ):
        self.inpaint_radius = inpaint_radius
        self.denoise_h = denoise_h
        self.clahe_clip = clahe_clip
        self.apply_denoise_and_clahe = apply_denoise_and_clahe
        self.last_info: Dict[str, object] = {}

    def restore(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Restores an H x W x 3 uint8 RGB image using classical OpenCV operations."""
        if mask is None:
            mask_u8 = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask_u8 = ((mask > 0).astype(np.uint8)) * 255

        mask_ratio = float(np.mean(mask_u8 > 0))
        low_conf = mask_ratio > LARGE_MISSING_THRESHOLD

        # Corner Case (Section 5.4 & 8.1): Clean image with no damage stays untouched
        if mask_ratio == 0.0:
            self.last_info = {
                "mode": "opencv_baseline",
                "mask_ratio": 0.0,
                "low_confidence": False,
                "note": "no damage detected; original image preserved",
            }
            return image.copy()

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Step 1: Telea Fast Marching Inpainting on the damage mask
        inpainted_bgr = cv2.inpaint(
            bgr, mask_u8, inpaintRadius=self.inpaint_radius, flags=cv2.INPAINT_TELEA
        )

        if self.apply_denoise_and_clahe:
            # Step 2: Non-Local Means denoising (Section 3.5 item 1)
            denoised_bgr = cv2.fastNlMeansDenoisingColored(
                inpainted_bgr, None, self.denoise_h, self.denoise_h, 7, 21
            )
            # Step 3: CLAHE contrast enhancement in CIE LAB space (Section 3.5 item 1)
            lab = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            out_bgr = inpainted_bgr

        repaired_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        self.last_info = {
            "mode": "opencv_baseline",
            "mask_ratio": mask_ratio,
            "low_confidence": low_conf,
            "note": (
                "warning: large missing region (>20% masked), low confidence"
                if low_conf
                else "Telea inpainting + NLM denoising + CLAHE completed"
            ),
        }
        return repaired_rgb


# ==============================================================================
# 2. Wan et al. (2020) Dual-VAE + Partial Non-Local Latent Mapping Architecture
# ==============================================================================


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation for stable latent features."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                padding_mode="reflect",
            ),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class VAEEncoder(nn.Module):
    """Variational Autoencoder Encoder (Wan et al., 2020).

    Compresses an RGB image X in R^{B x 3 x H x W} through progressive convolutional layers
    into a compact spatial latent distribution N(mu, sigma^2) at resolution (H/4, W/4),
    plus multi-scale structural skip features for crisp reconstruction of undamaged pixels.
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 32, latent_ch: int = 128):
        super().__init__()
        # Level 1: Full resolution (H, W) -> extracts low-level edges, corners, fine grain
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(inplace=True),
            ResBlock(base_ch),
        )
        # Level 2: Downsample 2x (H/2, W/2) -> extracts mid-level textures and contours
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(inplace=True),
            ResBlock(base_ch * 2),
        )
        # Level 3: Downsample 4x (H/4, W/4) -> extracts high-level shapes and semantic patterns
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, latent_ch, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, latent_ch),
            nn.SiLU(inplace=True),
            ResBlock(latent_ch),
        )
        # Gaussian posterior heads: mean (mu) and log-variance (logvar)
        self.conv_mu = nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv_logvar = nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1, padding_mode="reflect")

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes input image into latent code z, mu, logvar, and multi-scale skips."""
        f1 = self.stem(x)       # (B, base_ch, H, W)
        f2 = self.down1(f1)     # (B, base_ch*2, H/2, W/2)
        h = self.down2(f2)      # (B, latent_ch, H/4, W/4)

        mu = self.conv_mu(h)
        logvar = torch.clamp(self.conv_logvar(h), -10.0, 2.0)

        if deterministic or not self.training:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

        return z, mu, logvar, (f1, f2)


class VAEDecoder(nn.Module):
    """Variational Autoencoder Decoder (Wan et al., 2020).

    Upsamples the restored latent representation z_Y in R^{B x C_z x H/4 x W/4}
    back to full spatial resolution (H, W) with skip connections.
    """

    def __init__(self, out_channels: int = 3, base_ch: int = 32, latent_ch: int = 128):
        super().__init__()
        self.pre = ResBlock(latent_ch)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(latent_ch, base_ch * 2, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(inplace=True),
            ResBlock(base_ch * 2),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(inplace=True),
            ResBlock(base_ch),
        )

        self.head = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        skips: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstructs full-resolution RGB image from latent code z and skip features."""
        f1, f2 = skips
        m2 = F.interpolate(mask, size=f2.shape[2:], mode="bilinear", align_corners=False)
        m1 = F.interpolate(mask, size=f1.shape[2:], mode="bilinear", align_corners=False)

        x = self.pre(z)
        x = self.up1(x)
        if x.shape[2:] != f2.shape[2:]:
            x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=False)
        clean_f2 = f2 * (1.0 - m2) + x * m2
        x = self.fuse1(torch.cat([x, clean_f2], dim=1))

        x = self.up2(x)
        if x.shape[2:] != f1.shape[2:]:
            x = F.interpolate(x, size=f1.shape[2:], mode="bilinear", align_corners=False)
        clean_f1 = f1 * (1.0 - m1) + x * m1
        x = self.fuse2(torch.cat([x, clean_f1], dim=1))

        return self.head(x)


class PartialNonLocalBlock(nn.Module):
    """Mask-guided Partial Non-Local Block in Latent Space (Wan et al., 2020, Section 3.5).

    Copies clean texture features strictly from undamaged latent positions (M_latent == 0)
    into damaged/scratched latent positions (M_latent == 1) via masked self-attention.
    """

    def __init__(self, channels: int = 128, reduction: int = 2):
        super().__init__()
        mid_ch = max(16, channels // reduction)
        self.theta = nn.Conv2d(channels, mid_ch, kernel_size=1)
        self.phi = nn.Conv2d(channels, mid_ch, kernel_size=1)
        self.g = nn.Conv2d(channels, mid_ch, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_ch, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, z: torch.Tensor, latent_mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape
        pool_stride = 2 if max(h, w) > 32 else 1
        if pool_stride > 1:
            z_kv = F.avg_pool2d(z, kernel_size=pool_stride, stride=pool_stride)
            m_kv = F.avg_pool2d(latent_mask, kernel_size=pool_stride, stride=pool_stride)
        else:
            z_kv = z
            m_kv = latent_mask

        q = self.theta(z).view(b, -1, h * w).permute(0, 2, 1)  # (B, HW, C')
        k = self.phi(z_kv).view(b, -1, z_kv.shape[2] * z_kv.shape[3])  # (B, C', N_kv)
        v = self.g(z_kv).view(b, -1, z_kv.shape[2] * z_kv.shape[3]).permute(0, 2, 1)  # (B, N_kv, C')

        scale = q.shape[-1] ** -0.5
        attn = torch.bmm(q, k) * scale

        valid_keys = (m_kv.view(b, 1, -1) < 0.5).float()
        all_damaged = (valid_keys.sum(dim=-1, keepdim=True) == 0).float()
        valid_keys = valid_keys + all_damaged

        attn = attn.masked_fill(valid_keys == 0, -1e4)
        attn = F.softmax(attn, dim=-1)

        context = torch.bmm(attn, v).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.out_proj(context)
        return z + latent_mask * context


class LatentMappingNetwork(nn.Module):
    """Latent Mapping Network T_{Z_X -> Z_Y} (Wan et al., 2020, Section 3.4 & 3.5).

    Translates the damaged latent representation Z_X into the clean latent representation Z_Y.
    Also includes a high-frequency spatial texture refinement head that restores fine grain
    and edge continuity across the masked region. During fine-tuning, both VAEs are frozen
    and ONLY this mapping network trains.
    """

    def __init__(self, latent_ch: int = 128, base_ch: int = 32):
        super().__init__()
        self.mask_embed = nn.Sequential(
            nn.Conv2d(latent_ch + 1, latent_ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, latent_ch),
            nn.SiLU(inplace=True),
        )
        self.res1 = ResBlock(latent_ch, dilation=1)
        self.res2 = ResBlock(latent_ch, dilation=2)
        self.partial_non_local = PartialNonLocalBlock(latent_ch)
        self.res3 = ResBlock(latent_ch, dilation=4)
        self.res4 = ResBlock(latent_ch, dilation=1)

        self.gate = nn.Sequential(
            nn.Conv2d(latent_ch + 1, latent_ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Conv2d(latent_ch, latent_ch, kernel_size=3, padding=1, padding_mode="reflect")

        # Fine-scale context propagator in the mapping module (operates on masked image + latent guidance)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(4, base_ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(inplace=True),
            ResBlock(base_ch, dilation=1),
            ResBlock(base_ch, dilation=2),
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Tanh(),
        )
        # Initialize spatial residual refinement near zero for zero-shot stability
        nn.init.zeros_(self.spatial_refine[-2].weight)
        nn.init.zeros_(self.spatial_refine[-2].bias)

    def forward(self, z_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        latent_mask = F.interpolate(mask, size=z_x.shape[2:], mode="bilinear", align_corners=False)
        h = self.mask_embed(torch.cat([z_x, latent_mask], dim=1))
        h = self.res1(h)
        h = self.res2(h)
        h = self.partial_non_local(h, latent_mask)
        h = self.res3(h)
        h = self.res4(h)

        g = self.gate(torch.cat([h, latent_mask], dim=1))
        z_y = z_x + (self.out_conv(h) * g) * latent_mask
        return z_y


def _multi_scale_hole_fill(damaged: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Propagates clean surrounding pixels smoothly inward across masked holes on GPU/CPU."""
    x = damaged * (1.0 - mask)
    valid = 1.0 - mask
    for _ in range(4):
        x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
        v_pad = F.pad(valid, (2, 2, 2, 2), mode="reflect")
        x_sum = F.avg_pool2d(x_pad, kernel_size=5, stride=1) * 25.0
        v_sum = F.avg_pool2d(v_pad, kernel_size=5, stride=1) * 25.0
        filled = x_sum / torch.clamp(v_sum, min=1e-4)
        x = x * valid + filled * (1.0 - valid)
        valid = torch.clamp(v_sum, max=1.0)
    return damaged * (1.0 - mask) + x * mask


class WanLatentVAE(nn.Module):
    """Complete Dual-VAE Latent-Space Restoration Network (Wan et al., 2020)."""

    def __init__(self, base_ch: int = 32, latent_ch: int = 128):
        super().__init__()
        self.vae1_encoder = VAEEncoder(in_channels=3, base_ch=base_ch, latent_ch=latent_ch)
        self.vae2_encoder = VAEEncoder(in_channels=3, base_ch=base_ch, latent_ch=latent_ch)
        self.mapping_net = LatentMappingNetwork(latent_ch=latent_ch, base_ch=base_ch)
        self.vae2_decoder = VAEDecoder(out_channels=3, base_ch=base_ch, latent_ch=latent_ch)

    def freeze_vaes(self) -> None:
        """Freezes both VAEs so fine-tuning updates only the Latent Mapping Network (Section 3.4)."""
        for module in (self.vae1_encoder, self.vae2_encoder, self.vae2_decoder):
            for param in module.parameters():
                param.requires_grad = False
        for param in self.mapping_net.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreezes all parameters (used during initial VAE warm-up / pretraining)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        damaged: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: Damaged photo X + Mask M -> Latent Z_X -> Mapped Z_Y -> Restored Y_hat."""
        init_filled = _multi_scale_hole_fill(damaged, mask)
        z_x, mu_x, logvar_x, skips = self.vae1_encoder(init_filled, deterministic=deterministic)
        z_y = self.mapping_net(z_x, mask)
        vae_recon = self.vae2_decoder(z_y, skips, mask)

        # Combine VAE decoded structure with the mapping network's spatial texture refinement
        delta = self.mapping_net.spatial_refine(torch.cat([vae_recon, mask], dim=1))
        pred_raw = torch.clamp(vae_recon + 0.35 * delta, 0.0, 1.0)

        # Mask-guided composite blending: preserve 100% of undamaged pixels outside mask
        pred_comp = mask * pred_raw + (1.0 - mask) * damaged
        return {
            "pred": pred_comp,
            "pred_raw": pred_raw,
            "vae_recon": vae_recon,
            "z_x": z_x,
            "z_y": z_y,
            "mu_x": mu_x,
            "logvar_x": logvar_x,
        }


# ==============================================================================
# 3. Losses: L1 (Hole + Valid) + Perceptual Feature Loss + KL Divergence
# ==============================================================================


class PerceptualFeatureLoss(nn.Module):
    """Multi-scale perceptual & structural gradient loss (Section 3.4 & 8.2)."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
        ).unsqueeze(0)
        laplacian = torch.tensor(
            [[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]
        ).unsqueeze(0)
        filters = torch.cat([sobel_x, sobel_y, laplacian], dim=0)
        self.register_buffer("filters", filters.repeat(3, 1, 1, 1))

    def _extract_features(self, img: torch.Tensor) -> torch.Tensor:
        return F.conv2d(img, self.filters, padding=1, groups=3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(self._extract_features(pred), self._extract_features(target))
        pred_half = F.avg_pool2d(pred, 2)
        target_half = F.avg_pool2d(target, 2)
        loss = loss + 0.5 * F.l1_loss(
            self._extract_features(pred_half), self._extract_features(target_half)
        )
        return loss


class WanRepairLoss(nn.Module):
    """Compound training objective for Wan et al. (2020) Latent-Space Repair (Section 3.4)."""

    def __init__(
        self,
        hole_weight: float = 8.0,
        valid_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        latent_weight: float = 0.5,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.perceptual_weight = perceptual_weight
        self.latent_weight = latent_weight
        self.kl_weight = kl_weight
        self.perceptual = PerceptualFeatureLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        mask: torch.Tensor,
        z_clean_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_raw = outputs["pred_raw"]
        pred_comp = outputs["pred"]
        mu = outputs["mu_x"]
        logvar = outputs["logvar_x"]

        hole_l1 = torch.mean(torch.abs(mask * (pred_raw - clean))) / (
            torch.mean(mask) + 1e-4
        )
        valid_l1 = torch.mean(torch.abs((1.0 - mask) * (pred_raw - clean))) / (
            torch.mean(1.0 - mask) + 1e-4
        )
        perc_loss = self.perceptual(pred_comp, clean)
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        total = (
            self.hole_weight * hole_l1
            + self.valid_weight * valid_l1
            + self.perceptual_weight * perc_loss
            + self.kl_weight * kl_loss
        )

        latent_l1_val = 0.0
        if z_clean_target is not None:
            latent_l1 = F.l1_loss(outputs["z_y"], z_clean_target.detach())
            total = total + self.latent_weight * latent_l1
            latent_l1_val = float(latent_l1.item())

        return total, {
            "total": float(total.item()),
            "hole_l1": float(hole_l1.item()),
            "valid_l1": float(valid_l1.item()),
            "perceptual": float(perc_loss.item()),
            "latent_l1": latent_l1_val,
            "kl": float(kl_loss.item()),
        }


# ==============================================================================
# 4. Production Restorer Wrapper (Tiled Inference + Feathered Mask Blending)
# ==============================================================================


class WanLatentRestorer:
    """High-level Restorer wrapping WanLatentVAE for Zero-Shot and Fine-Tuned inference."""

    def __init__(
        self,
        model: Optional[WanLatentVAE] = None,
        checkpoint_path: Optional[Path | str] = DEFAULT_REPAIR_CKPT,
        mode: str = "fine_tuned",
        device: Optional[str] = None,
        tile_size: int = 256,
        overlap: int = 32,
        dilate_radius: int = 0,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.mode = mode
        self.tile_size = tile_size
        self.overlap = overlap
        self.dilate_radius = dilate_radius
        self.last_info: Dict[str, object] = {}

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = WanLatentVAE().to(self.device)
            if checkpoint_path is not None and Path(checkpoint_path).exists():
                ckpt = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=True
                )
                if isinstance(ckpt, dict) and mode in ckpt:
                    self.model.load_state_dict(ckpt[mode])
                elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["state_dict"])
                else:
                    self.model.load_state_dict(ckpt)
        self.model.eval()

    @torch.no_grad()
    def restore(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Restores an H x W x 3 uint8 RGB image using the Wan et al. Latent-Space VAE."""
        if mask is None:
            mask_bin = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask_bin = (mask > 0).astype(np.uint8)

        mask_ratio = float(np.mean(mask_bin > 0))
        low_conf = mask_ratio > LARGE_MISSING_THRESHOLD

        if mask_ratio == 0.0:
            self.last_info = {
                "mode": self.mode,
                "mask_ratio": 0.0,
                "low_confidence": False,
                "note": "no damage detected; original image preserved",
            }
            return image.copy()

        if self.dilate_radius > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.dilate_radius + 1, 2 * self.dilate_radius + 1),
            )
            mask_dil = cv2.dilate(mask_bin, k, iterations=1)
        else:
            mask_dil = mask_bin

        orig_f = image.astype(np.float32) / 255.0
        mask_f = mask_dil.astype(np.float32)

        h, w = orig_f.shape[:2]
        ts = self.tile_size
        stride = max(1, ts - self.overlap)

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        weights = np.zeros((h, w, 1), dtype=np.float32)

        win_y = np.hanning(ts + 2)[1:-1]
        win_x = np.hanning(ts + 2)[1:-1]
        window = np.outer(win_y, win_x).astype(np.float32)[..., None]

        y_points = list(range(0, max(1, h - ts + 1), stride))
        if h > ts and y_points[-1] + ts < h:
            y_points.append(h - ts)
        x_points = list(range(0, max(1, w - ts + 1), stride))
        if w > ts and x_points[-1] + ts < w:
            x_points.append(w - ts)

        for y in y_points:
            for x in x_points:
                patch_img = orig_f[y : y + ts, x : x + ts]
                patch_mask = mask_f[y : y + ts, x : x + ts]
                ph, pw = patch_img.shape[:2]

                pad_h = max(0, ts - ph) if (h >= ts or ph < ts) else (4 - ph % 4) % 4
                pad_w = max(0, ts - pw) if (w >= ts or pw < ts) else (4 - pw % 4) % 4

                if pad_h > 0 or pad_w > 0:
                    patch_img_p = np.pad(
                        patch_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
                    )
                    patch_mask_p = np.pad(
                        patch_mask, ((0, pad_h), (0, pad_w)), mode="reflect"
                    )
                else:
                    patch_img_p = patch_img
                    patch_mask_p = patch_mask

                if np.all(patch_mask_p == 0):
                    pred_patch = patch_img
                else:
                    t_img = (
                        torch.from_numpy(patch_img_p)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    t_mask = (
                        torch.from_numpy(patch_mask_p)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    out_dict = self.model(t_img, t_mask, deterministic=True)
                    pred_full = (
                        out_dict["pred_raw"]
                        .squeeze(0)
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    pred_patch = pred_full[:ph, :pw]

                w_patch = window[:ph, :pw]
                canvas[y : y + ph, x : x + pw] += pred_patch * w_patch
                weights[y : y + ph, x : x + pw] += w_patch

        pred_reconstructed = np.clip(canvas / np.maximum(weights, 1e-6), 0.0, 1.0)

        # Keep full 1.0 replacement weight on the damaged mask core so 0% of the scratch leaks back,
        # while feathering only the outer 1-px transition boundary.
        feathered_rim = cv2.GaussianBlur(mask_f, (3, 3), sigmaX=0.7)
        soft_mask = np.maximum(mask_f, 0.35 * feathered_rim)[..., None]
        final_f = soft_mask * pred_reconstructed + (1.0 - soft_mask) * orig_f
        final_u8 = np.clip(np.round(final_f * 255.0), 0, 255).astype(np.uint8)

        self.last_info = {
            "mode": self.mode,
            "mask_ratio": mask_ratio,
            "low_confidence": low_conf,
            "note": (
                "warning: large missing region (>20% masked), low confidence"
                if low_conf
                else f"Wan et al. latent VAE ({self.mode}) restoration completed"
            ),
        }
        return final_u8


Restorer = WanLatentRestorer

__all__ = [
    "LARGE_MISSING_THRESHOLD",
    "OpenCVBaselineRestorer",
    "VAEEncoder",
    "VAEDecoder",
    "PartialNonLocalBlock",
    "LatentMappingNetwork",
    "WanLatentVAE",
    "PerceptualFeatureLoss",
    "WanRepairLoss",
    "WanLatentRestorer",
    "Restorer",
]
