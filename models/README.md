# Models Directory

This folder stores trained model weights and checkpoints for the restoration pipeline.

## Available Checkpoints

### `damage_detection_unet.pt` (Stage 2: Physical Damage Detection)
* **File Size:** 7.84 MB (1,948,289 float32 parameters)
* **Architecture:** 4-level dense U-Net with skip connections
* **Input / Output:** `(B, 3, H, W)` RGB image $\to$ `(B, 1, H, W)` defect segmentation logits
* **Training Details:**
  * Runtime: Tesla T4 GPU (Google Cloud VM via Colab)
  * Scale: 15 epochs across 3,000 balanced image pairs (1,500 real ARTeFACT tiles + 1,500 synthetic defect pairs)
  * Loss Function: Compound Loss (Binary Cross-Entropy + Soft Dice Loss, 1:1 weighting)
  * Optimizer: AdamW ($3 \times 10^{-4} \to 1 \times 10^{-5}$ via Cosine Annealing schedule)
* **Convergence Metrics:**
  * Validation IoU: $43.19\%$ (Peak overlap on physical cracks/scratches)
  * Validation F1 (Dice): $49.34\%$
  * Overall Pixel Accuracy: $\sim 97.4\%$
* **Loading in Python / PyTorch:**
  ```python
  import torch

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint_path = "models/damage_detection_unet.pt"

  # Initialize architecture (defined in notebooks/02_damage_detection or src/)
  model = DamageDetectionUNet(in_channels=3, out_channels=1)
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))
  model.to(device)
  model.eval()
  ```
* **Downstream Integration (photo repair, notebook 03):**
  Outputs continuous defect probability maps $\sigma(Z) \in [0.0, 1.0]$. Applying a threshold ($\tau \approx 0.45\text{--}0.50$) produces binary defect masks that guide inpainting (e.g., Partial Convolution / LaMa) on damaged regions.

---

### `repair_wan_mapping_finetuned.part1of2.pt` + `.part2of2.pt` (photo repair, notebook 03)
* **What it is:** the latent mapping network of Wan et al. (2020), *Bringing Old Photos Back to Life* (Microsoft's published `mapping_scratch` model, release v1.0), fine-tuned on OpenPhoto Restore. Only the mapping network is stored. The two VAEs (`VAE_A_quality`, `VAE_B_scratch`) stay frozen at Microsoft's published weights, which `src/repair.py:setup_wan_repo` downloads at run time.
* **Files:** 51.3M parameters in fp16, split into two shards so each stays under GitHub's 100 MB limit (85.5 MB + 17.3 MB). `load_mapping_weights` joins them.
* **Training:** Colab L4. 5 epochs per seed (seeds 42, 123, 999), batch 8, random 256 x 256 crops, Adam (lr 1e-4). Each seed starts from the published mapping weights. Loss: 60 x L1 on the mapped latent vs VAE_B(clean) + 10 x image L1, following Microsoft's `train_mapping.py` minus the GAN and VGG terms. Masks come from `damage_detection_unet.pt`.
* **Data:** OpenPhoto Restore, split by source photo into train 4,050 / val 450 / test 497. The best epoch was chosen on validation masked PSNR; the shipped seed is **123** (val masked PSNR_Y 22.30 dB; test PSNR_Y 23.84 dB).
* **Test results** (497 held-out photos, luminance; full table in `results/tables/s2_repair_ablation.csv`):

  | Condition | PSNR_Y | SSIM_Y | LPIPS | PSNR_Y inside damage |
  |---|---|---|---|---|
  | Input (no repair) | 21.65 | 0.8551 | 0.1719 | 16.73 |
  | Classical (inpaint + denoise + contrast) | 20.12 | 0.8248 | 0.2253 | 16.00 |
  | Pretrained Wan et al. (2020) | 21.76 | 0.8058 | 0.2092 | 17.06 |
  | Fine-tuned Wan et al. (this checkpoint's recipe, 3 seeds) | 23.89 ± 0.05 | 0.8276 ± 0.0034 | 0.1983 ± 0.0030 | 20.60 ± 0.06 |

* **Loading:**
  ```python
  from src.repair import setup_wan_repo, load_wan_scratch_model, load_mapping_weights, WanRestorer

  model = load_wan_scratch_model(setup_wan_repo("data/work/Bringing-Old-Photos-Back-to-Life"))
  load_mapping_weights(model, "models/repair_wan_mapping_finetuned.pt")   # reads both shards
  repaired_rgb = WanRestorer(model, mode="fine_tuned").restore(damaged_rgb, damage_mask)  # H x W x 3 uint8
  ```

  Masks for this table come from the notebook-02 detector; "inside damage" is scored on reference damage masks. This model is a baseline: the shipped repair method is the tone-correction + LaMa pipeline below.

---

### `damage_detection_unet_openphoto.pt` (photo repair, notebook 03)
* **What it is:** a copy of `damage_detection_unet.pt` (Sai Krishna Akula's notebook-02 detector, same architecture) fine-tuned on OpenPhoto Restore. The original checkpoint and notebook 02 are unchanged.
* **Why:** on OpenPhoto the notebook-02 detector finds 4% of the damage (recall 0.039, IoU 0.034), so local repair had almost nothing to work with.
* **Training:** reference masks derived from each training pair (`src/reference_masks.py`); BCE (with positive weighting) + Dice loss; 256 x 256 crops, 8,000 crops per epoch, batch 16, AdamW 1e-3 with OneCycle, 12 epochs on a Colab L4; best epoch by validation IoU (epoch 12). Code: `src/detector_finetune.py`.
* **Threshold:** 0.7, chosen on validation from {0.4, 0.5, 0.6, 0.7}.
* **Test results** (497 photos, against reference masks): recall 0.780, precision 0.547, IoU 0.474, mean coverage 5.0% of pixels.
* **Loading:** same as `damage_detection_unet.pt`; apply a 0.7 threshold to the sigmoid output.

---

### `repair_tone_curve.pt` (photo repair, notebook 03)
* **What it is:** a small CNN (`ToneCurveNet`, about 0.6M parameters, stored in fp16) that predicts a monotone 16-knot tone curve from 128 x 128 thumbnails of the photo's luminance and saturation. The curve is applied to the full-resolution luminance, which removes fading, sepia casts and contrast loss without touching structure.
* **Training:** OpenPhoto Restore train split; L1 loss against the clean luminance; AdamW with cosine schedule, 40 epochs, seeds 42, 123 and 999. Shipped seed **42** (validation PSNR_Y 28.32 dB). Code: `src/tone.py`.
* **Test results** (497 photos): tone correction alone 27.89 ± 0.08 dB PSNR_Y over 3 seeds. As part of the full pipeline (adapted detector + tone + LaMa): **28.73 dB PSNR_Y, SSIM 0.8862, LPIPS 0.1325**, 21.56 dB inside the damage.
* **Loading:**
  ```python
  from src.tone import load_tone_net, ToneCorrector
  from src.inpaint import LamaInpainter, RepairPipeline

  tone = ToneCorrector(load_tone_net("models/repair_tone_curve.pt", device))
  lama = LamaInpainter("data/work/big-lama.pt", device)   # downloaded on first use
  pipeline = RepairPipeline(adapted_detector, tone, lama)
  repaired_rgb = pipeline.restore(damaged_rgb)             # H x W x 3 uint8 (grayscale)
  ```
