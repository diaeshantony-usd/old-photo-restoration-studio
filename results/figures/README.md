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
* **Contents:** 2 x 7 panels on test photos #251 and #143, two of the eight best-case photos in `repair_pipeline_examples.png`, both with large cracks. They were picked with the help of the clean originals, so they show a complete repair, not the typical result.
* **Masks:** the baselines use the notebook-02 detector masks, as in their test-set scores; the full pipeline uses the OpenPhoto-adapted detector, whose mask is shown in the second column.
* **Numbers** (luminance PSNR vs the clean photo; whole image / inside the reference damage mask):

  | Photo | Damaged input | Classical | Pretrained Wan et al. | Fine-tuned Wan et al. | Full pipeline (ours) |
  |---|---|---|---|---|---|
  | #251 (8.0% masked) | 22.70 / 12.94 dB | 21.85 / 11.96 dB | 21.43 / 11.56 dB | 28.48 / 28.89 dB | 26.53 / 28.34 dB |
  | #143 (9.9% masked) | 16.61 / 15.35 dB | 19.93 / 13.86 dB | 20.07 / 9.72 dB | 26.30 / 23.63 dB | **30.92** / **27.54** dB |

* **Panel columns:** damaged input, adapted detector mask, classical repair, pretrained Wan et al., fine-tuned Wan et al., full pipeline (adapted detector + tone correction + LaMa), clean original (grayscale, since repair is scored on luminance).

---

### `repair_pipeline_examples.png`
* **Stage:** photo repair (notebook `03_repair`)
* **Contents:** the eight test photos the full pipeline repairs most completely, each shown as damaged input, full-pipeline output and clean original, with whole-image PSNR (test photos #324, #205, #251, #311, #286, #329, #143, #140).
* **Selection rule:** among the 274 test photos with clearly visible local damage (reference mask covering at least 2% of the photo), each photo is repaired, a reference damage mask is derived again from the repaired photo and the clean original, and the eight with the least damage left (2.4% to 7.9% of the input's) are shown. The rule uses the clean originals, so these are **best cases**, not typical results; across all 274 photos the median damage left by this measure is 61% of the input's (the measure also counts inpainted pixels that differ from the original texture). The averages over all 497 test photos are in `results/tables/s2_repair_ablation.csv`.
