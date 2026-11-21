# tests

Run with `python -m pytest tests -q` from the repository root. These tests need no GPU and no model weights.

| File | Covers |
|---|---|
| `test_colorization.py` | Gray / sepia / color skip rule; a color photo is returned unchanged; colorization can be switched off |
| `test_enhancement_helpers.py` | 4096 px output cap; minimum face size filter |
| `test_metrics.py` | PSNR, IoU / F1, colorfulness; SSIM and Delta E when scikit-image is installed |
| `test_pipeline.py` | Stage order and switches, skipped stages are logged, face weight is forwarded, alpha flattened to RGB |

Checks that need the real models (a photo without faces skips GFPGAN, large image on a T4, deterministic output, tiny faces skipped) run as labelled cells in `notebooks/04_colorization` and `notebooks/05_enhancement`.
