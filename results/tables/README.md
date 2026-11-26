# tables

Result tables from the repair notebook (`notebooks/03_repair/03_repair.ipynb`). All scores are on the 497 held-out OpenPhoto Restore test photos at 512 x 512, measured on luminance (BT.601 Y) against the clean original.

## `s2_repair_ablation.csv`

One row per method, in the order of the notebook's results table. Methods trained with three seeds (42, 123 and 999) show mean ± sample standard deviation.

* `Method`: unrepaired input, classical repair (inpaint + denoise + contrast), classical repair on masked pixels only, pretrained Wan et al. (2020), fine-tuned Wan et al., learned tone correction, the full pipeline (adapted detector + tone correction + LaMa), and three comparisons using the adapted detector's masks (classical masked-only, fine-tuned Wan et al., tone correction + classical inpainting).
* `PSNR_Y_dB`, `SSIM_Y`: whole-photo PSNR and SSIM.
* `LPIPS_Y`: LPIPS (AlexNet, `lpips` package); lower is better.
* `MaskedPSNR_Y_dB`: PSNR inside the reference damage mask (derived from each damaged/clean pair, `src/reference_masks.py`), so every method is scored on the same pixels whatever mask it was given.
* `PSNR_RGB_dB`: RGB PSNR against the colour original, for reference only (OpenPhoto damage removes colour, which is the colorization notebook's job).
* `Runtime_ms_median`: median time per photo (OpenCV on CPU, neural models on a Colab L4 GPU).

## `s2_repair_per_image.csv`

One row per test photo and method; seeded methods appear once per seed. Columns: `id`, `psnr_y`, `ssim_y`, `lpips_y`, `masked_psnr_y`, `psnr_rgb`, `runtime_ms`, `method`. `masked_psnr_y` is empty when the reference mask is empty.

## `s2_repair_paired_vs_input.csv`

Per-photo difference of each method against the unrepaired input, for each metric, over all photos and over the 443 photos whose reference mask is not empty. `mean_gain_vs_input` is positive when the method is better (for LPIPS the sign is flipped so higher is better). `ci95_low` and `ci95_high` are a 5,000-sample paired bootstrap over photos; `share_improved` is the fraction of photos that got better.

## `s2_repair_components.csv`

What each part of the pipeline contributes, and how far it is from an upper bound. Rows: unrepaired input; tone correction only; tone + classical inpainting (adapted masks); tone + LaMa (adapted masks, the full pipeline); tone + LaMa with reference masks; perfect hole filling (clean pixels pasted into the reference mask, tone untouched); perfect tone correction (best possible global tone curve, no inpainting). Reference masks and the two upper bounds use the clean original, so they are analysis only, not methods the application can run. Columns: PSNR, SSIM, LPIPS and PSNR inside the reference mask (100 dB is the cap for identical pixels).
