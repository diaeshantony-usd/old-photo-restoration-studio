# Results — Diagnostic Figures

This directory holds exported diagnostic visualizations, inference figures, and quantitative performance audits for the project.

## Catalog of Figures

### `damage_detection_verification.png`
* **Stage:** Stage 2 (Physical Damage Detection)
* **Resolution:** $2640 \times 1080$ (10-panel dual-case visual audit)
* **Contents:**
  * **Case A: Real Archival Physical Damage (`cljmrkz5p345z07clgeqi8bns.png`)**
    * 17th/18th-century historical scene with physical paper tear and canvas boundary loss.
    * Metrics: **$45.1\%$ IoU**, peak defect confidence **$1.00$**.
    * Demonstrates accurate real-world edge damage boundary detection with green True Positive alignment.
  * **Case B: Hairline Scratches, Creases & Particulate Dust (`synth_00004.png`)**
    * Dense defect network on vintage photographic emulsion.
    * Metrics: **$62.1\%$ IoU**, peak defect confidence **$1.00$**.
    * Demonstrates sub-millimeter hairline crack detection without false positive artifacts across smooth photographic regions.
* **Panel Columns:**
  1. Degraded Input Photo ($X$)
  2. Ground-Truth Damage Mask ($Y$)
  3. Continuous Defect Probability Heatmap ($\hat{P}$)
  4. Predicted Binary Damage Mask ($\hat{Y}$)
  5. Color-Coded Error Overlay (Green: True Positive, Red: False Positive, Blue: False Negative)

---

### `repair_verification.png`
* **Stage:** photo repair (notebook `03_repair`)
* **Contents:** 2 x 7 panels on two photos from the held-out OpenPhoto Restore **test** split (test photos #303 and #476), chosen by a fixed seeded rule before any model was run.
* **Mask column:** the mask predicted by the OpenPhoto-adapted detector (`damage_detection_unet_openphoto.pt`), which the full pipeline uses.
* **Numbers** (luminance PSNR vs the clean photo; whole image / inside the reference damage mask):

  | Photo | Damaged input | Classical | Pretrained Wan et al. | Fine-tuned Wan et al. | Full pipeline (ours) |
  |---|---|---|---|---|---|
  | #303 (3.1% masked) | 27.31 / 14.07 dB | 23.36 / 13.95 dB | 21.45 / 13.25 dB | 23.27 / 18.04 dB | **29.71** / 17.41 dB |
  | #476 (5.4% masked) | 20.81 / 11.39 dB | 19.02 / 11.10 dB | 20.44 / 11.64 dB | 23.80 / 15.44 dB | **26.98** / **16.06** dB |

* **Panel columns:** damaged input, adapted detector mask, classical repair, pretrained Wan et al., fine-tuned Wan et al., full pipeline (adapted detector + tone correction + LaMa), clean original (grayscale, since repair is scored on luminance).

---

### `repair_pipeline_examples.png`
* **Stage:** photo repair (notebook `03_repair`)
* **Contents:** 8 test photos drawn at random (fixed seed), each shown as damaged input, full-pipeline output and clean original, with whole-image PSNR. Scratches, cracks and tears are removed on most photos; on #299 a wide tear is only partly filled, which shows the detector's main remaining weakness.
