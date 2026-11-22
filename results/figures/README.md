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
* **Stage:** Stage 3 (Defect Inpainting & Latent-Space Restoration — Wan et al., 2020)
* **Contents (2×5 Dual-Case Real Photograph Visual Audit):**
  * **Case A: Real Color Human Portrait (`astronaut`) with Facial & Background Scratches**
    * Degraded Input PSNR: **$20.57\text{ dB}$** ($3.1\%$ pixels flagged by Stage 2 mask)
    * Classical OpenCV Baseline (Telea + NLM + CLAHE): **$23.12\text{ dB}$**
    * Fine-Tuned Wan et al. (2020) Dual-VAE Restoration: **$33.02\text{ dB}$** (**$+9.90\text{ dB}$ gain over OpenCV baseline**)
  * **Case B: Real Vintage Monochrome Street Photograph (`camera`) with Creases & Dust**
    * Degraded Input PSNR: **$22.84\text{ dB}$** ($1.9\%$ pixels flagged by Stage 2 mask)
    * Classical OpenCV Baseline (Telea + NLM + CLAHE): **$25.83\text{ dB}$**
    * Fine-Tuned Wan et al. (2020) Dual-VAE Restoration: **$40.80\text{ dB}$** (**$+14.97\text{ dB}$ gain over OpenCV baseline**)
* **Panel Columns:**
  1. Degraded Input Real Photograph ($X$)
  2. Stage 2 Binary Damage Mask ($M$)
  3. Condition 1: Classical OpenCV Baseline (Telea + NLM + CLAHE)
  4. Condition 3: Fine-Tuned Wan et al. (2020) Latent-Space VAE ($\hat{Y}$)
  5. Clean Reference Real Photograph ($Y$)
