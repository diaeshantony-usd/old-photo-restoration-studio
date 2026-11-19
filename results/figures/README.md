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
