"""Learned tone correction for the repair stage (notebook 03).

OpenPhoto Restore's damage changes the tone of the whole photo (fading, contrast loss, sepia casts)
as well as adding local defects. One well-chosen brightness curve per photo removes most of that
error, but a fixed contrast stretch does not, because each photo needs a different curve. So the
curve is predicted from the damaged photo itself.

``ToneCurveNet`` looks at two small thumbnails of the damaged photo, its luminance and its colour
saturation, and predicts one monotone piecewise-linear curve with ``knots`` segments. The
saturation channel tells the network how strongly the photo has been faded or toned: a photo that
still has its colours is close to the original and needs little correction. The curve is applied to
the full-resolution luminance, so the correction cannot add or move details; it only remaps
brightness.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import to_luma

THUMB = 128     # network input size
LOSS_RES = 256  # resolution at which the training loss is computed


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class ToneCurveNet(nn.Module):
    """Predicts a monotone tone curve from N x 2 x 128 x 128 thumbnails (luminance, saturation; 0-1).

    The output is ``knots + 1`` curve values at evenly spaced input levels 0, 1/K, ..., 1.
    At initialisation the curve is the identity.
    """

    def __init__(self, knots: int = 16, in_ch: int = 2, hist_bins: int = 32):
        super().__init__()
        self.knots, self.hist_bins = knots, hist_bins
        self.features = nn.Sequential(_block(in_ch, 32), _block(32, 64), _block(64, 128), _block(128, 128))
        self.head = nn.Sequential(nn.Linear(128 + in_ch * hist_bins, 128), nn.ReLU(inplace=True), nn.Linear(128, knots + 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _histograms(self, x: torch.Tensor) -> torch.Tensor:
        idx = (x.clamp(0, 1) * (self.hist_bins - 1)).round().long().flatten(2)  # N x C x P
        h = torch.zeros(x.shape[0], x.shape[1], self.hist_bins, device=x.device)
        h.scatter_add_(2, idx, torch.ones_like(idx, dtype=h.dtype))
        return (h / idx.shape[-1]).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x).mean((2, 3))
        p = self.head(torch.cat([f, self._histograms(x)], 1))
        offset = 0.25 * torch.tanh(p[:, :1])                        # curve value at input 0
        steps = torch.exp(p[:, 1:].clamp(-6, 3)) / self.knots       # positive steps -> monotone curve
        return torch.cat([offset, offset + torch.cumsum(steps, 1)], 1)


def apply_curve(curve: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Applies per-image curves (N x K+1) to luminance images y (N x H x W, 0-1)."""
    k = curve.shape[1] - 1
    pos = y.clamp(0, 1) * k
    lo = pos.floor().clamp(max=k - 1).long()
    frac = pos - lo
    flat = lo.flatten(1)
    c0 = torch.gather(curve, 1, flat).view_as(y)
    c1 = torch.gather(curve, 1, flat + 1).view_as(y)
    return (c0 + frac * (c1 - c0)).clamp(0, 1)


def _resize(y: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(y, (size, size), interpolation=cv2.INTER_AREA)


def tone_features(damaged_rgb: np.ndarray):
    """Network input (2 x THUMB x THUMB: luminance, saturation) and full-resolution luminance, all 0-1."""
    y = to_luma(damaged_rgb).astype(np.float32) / 255.0
    saturation = cv2.cvtColor(damaged_rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    return np.stack([_resize(y, THUMB), _resize(saturation, THUMB)]), y


class ToneCorrector:
    """Applies a trained ToneCurveNet to a damaged photo. Returns an H x W x 3 uint8 gray image."""

    def __init__(self, net: ToneCurveNet, device="cuda"):
        self.net, self.device = net.to(device).eval(), torch.device(device)

    @torch.no_grad()
    def __call__(self, damaged_rgb: np.ndarray) -> np.ndarray:
        x, y = tone_features(damaged_rgb)
        curve = self.net(torch.from_numpy(x)[None].to(self.device))
        out = apply_curve(curve, torch.from_numpy(y)[None].to(self.device))[0].cpu().numpy()
        return np.repeat((out * 255 + 0.5).astype(np.uint8)[..., None], 3, axis=2)


def build_tone_dataset(damaged, clean) -> Dict[str, np.ndarray]:
    """Network inputs and loss-resolution luminance for N (damaged, clean) RGB uint8 pairs."""
    n = len(damaged)
    x = np.empty((n, 2, THUMB, THUMB), np.float32)
    src = np.empty((n, LOSS_RES, LOSS_RES), np.float32)
    tgt = np.empty((n, LOSS_RES, LOSS_RES), np.float32)
    for i in range(n):
        x[i], y = tone_features(damaged[i])
        src[i] = _resize(y, LOSS_RES)
        tgt[i] = _resize(to_luma(clean[i]).astype(np.float32) / 255.0, LOSS_RES)
    return {"x": x, "src": src, "tgt": tgt}


def train_tone_net(train: Dict[str, np.ndarray], val: Dict[str, np.ndarray], epochs: int = 40, batch: int = 64,
                   lr: float = 1e-3, seed: int = 0, device="cuda", log: Optional[Callable[[str], None]] = print):
    """Trains ToneCurveNet with an L1 loss on luminance; keeps the epoch with the best validation PSNR."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    net = ToneCurveNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    T = {k: torch.from_numpy(v) for k, v in train.items()}
    V = {k: torch.from_numpy(v).to(dev) for k, v in val.items()}

    def val_psnr() -> float:
        net.eval()
        with torch.no_grad():
            out = torch.cat([apply_curve(net(V["x"][i:i + 128]), V["src"][i:i + 128]) for i in range(0, len(V["x"]), 128)])
            mse = ((out - V["tgt"]) ** 2).mean((1, 2)).clamp_min(1e-10)
        return float((10 * torch.log10(1 / mse)).mean())

    best = (val_psnr(), copy.deepcopy(net.state_dict()), 0)
    history = [{"epoch": 0, "val_psnr": best[0]}]
    for ep in range(1, epochs + 1):
        net.train()
        perm = rng.permutation(len(T["x"]))
        total = 0.0
        for i in range(0, len(perm) - batch + 1, batch):
            idx = torch.from_numpy(perm[i:i + batch])
            x, src, tgt = T["x"][idx].to(dev), T["src"][idx].to(dev), T["tgt"][idx].to(dev)
            if rng.random() < 0.5:
                x, src, tgt = x.flip(-1), src.flip(-1), tgt.flip(-1)
            loss = F.l1_loss(apply_curve(net(x), src), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        v = val_psnr()
        history.append({"epoch": ep, "train_l1": total / len(perm), "val_psnr": v})
        if v > best[0]:
            best = (v, copy.deepcopy(net.state_dict()), ep)
        if log and (ep % 10 == 0 or ep == epochs):
            log(f"Tone epoch {ep:2d}: train L1 {total / len(perm):.4f} | validation PSNR {v:.2f} dB")
    net.load_state_dict(best[1])
    return net.eval(), {"best_val_psnr": best[0], "best_epoch": best[2], "history": history}


def save_tone_net(net: ToneCurveNet, path, meta: Optional[Dict] = None) -> None:
    torch.save({"state_dict": {k: v.half() for k, v in net.state_dict().items()}, "meta": meta or {}}, path)


def load_tone_net(path, device="cuda") -> ToneCurveNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ToneCurveNet()
    net.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    return net.to(device).eval()


__all__ = ["ToneCurveNet", "ToneCorrector", "apply_curve", "tone_features", "build_tone_dataset", "train_tone_net",
           "save_tone_net", "load_tone_net"]
