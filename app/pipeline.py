"""run_pipeline(): the one function the notebooks, the tests and the Gradio demo all call.

Stages run in this order, and each can be switched off: detect damage, repair, colorize, enhance.
Colorization and enhancement are implemented in ``src/colorization.py`` and ``src/enhancement.py``.
The detector and the repair model belong to other team members: plug them in as objects with

    detector.predict(image) -> H x W uint8 mask (1 = damaged)
    restorer.restore(image, mask) -> H x W x 3 uint8 image

If a requested stage has no implementation yet it is skipped and the reason is logged in
``result["log"]``, so the demo always shows exactly what ran.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.colorization import is_grayscale  # noqa: E402

ALL_STAGES = ("detect", "repair", "colorize", "enhance")


def prepare_image(img: Image.Image) -> np.ndarray:
    """PIL image -> RGB uint8 array. Applies EXIF rotation and flattens alpha on white."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return np.array(img.convert("RGB"))


class Pipeline:
    """Holds the stage objects. Pass ``None`` for a stage that is not available yet."""

    def __init__(self, detector=None, restorer=None, colorizer=None, enhancer=None):
        self.detector, self.restorer = detector, restorer
        self.colorizer, self.enhancer = colorizer, enhancer

    def run(self, image: np.ndarray, stages: Sequence[str] = ALL_STAGES,
            face_weight: Optional[float] = None) -> Dict:
        """Run the enabled stages in order and return every intermediate image plus a log."""
        out: Dict = {"input": image, "mask": None, "log": []}
        log = out["log"]
        mask = None

        def missing(name: str) -> None:
            log.append(f"{name}: skipped (not available yet)")

        if "detect" in stages:
            if self.detector is None:
                missing("detect")
            else:
                mask = self.detector.predict(image)
                out["mask"] = mask
                log.append(f"detect: {100 * float(np.mean(mask > 0)):.2f}% of pixels flagged")

        if "repair" in stages:
            if self.restorer is None:
                missing("repair")
            else:
                image = self.restorer.restore(image, mask if mask is not None else np.zeros(image.shape[:2], np.uint8))
                log.append("repair: done")
        out["repaired"] = image

        if "colorize" in stages:
            if self.colorizer is None:
                missing("colorize")
            else:
                image = self.colorizer.colorize(image)
                info = self.colorizer.last_info
                log.append(f"colorize: {'ran' if info['ran'] else 'skipped'} ({info['reason']}); "
                           "colors are a plausible guess, not the true colors" if info["ran"]
                           else f"colorize: skipped ({info['reason']})")
        out["colorized"] = image

        if "enhance" in stages:
            if self.enhancer is None:
                missing("enhance")
            else:
                if face_weight is not None:
                    self.enhancer.face_weight = face_weight
                image = self.enhancer.enhance(image)
                info = self.enhancer.last_info
                log.append(f"enhance: {info.get('face_step', info.get('reason', ''))}; "
                           f"scale {info.get('outscale', 1):.2f}x; faces are AI-restored" if info.get("ran")
                           else f"enhance: skipped ({info.get('reason', '')})")
        out["output"] = image
        return out


_default: Optional[Pipeline] = None


def get_default_pipeline() -> Pipeline:
    """Build the default pipeline once: real colorizer and enhancer, detector/repair when available."""
    global _default
    if _default is None:
        from src.colorization import Colorizer
        from src.enhancement import Enhancer, load_models

        up, gfp = load_models(model_dir=str(ROOT / "models"))
        _default = Pipeline(colorizer=Colorizer(), enhancer=Enhancer(up, gfp))
    return _default


def run_pipeline(image: np.ndarray, stages: Sequence[str] = ALL_STAGES, **kwargs) -> Dict:
    return get_default_pipeline().run(image, stages, **kwargs)


__all__ = ["ALL_STAGES", "Pipeline", "prepare_image", "run_pipeline", "get_default_pipeline", "is_grayscale"]
