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
* **Downstream Integration (Stage 3):**
  Outputs continuous defect probability maps $\sigma(Z) \in [0.0, 1.0]$. Applying a threshold ($\tau \approx 0.45\text{--}0.50$) produces binary defect masks that guide inpainting (e.g., Partial Convolution / LaMa) on damaged regions.

---

### `repair_wan_latent_vae.pt` (Stage 3: Defect Inpainting & Latent-Space Restoration)
* **Architecture:** Wan et al. (2020) Dual-VAE Latent-Space Restoration (`WanLatentVAE`, 3,972,838 total parameters: 2,272,099 frozen $\text{VAE}_1 / \text{VAE}_2$ parameters + 1,700,739 trainable `LatentMappingNetwork` parameters with a `PartialNonLocalBlock`)
* **Input / Output:** `(B, 3, H, W)` Damaged RGB image + `(B, 1, H, W)` Stage 2 binary damage mask $\to$ `(B, 3, H, W)` Repaired RGB image
* **Training & Fine-Tuning Details:**
  * Runtime: Tesla T4 GPU (Google Colab, mixed precision `fp16`)
  * Protocol (Design Document Section 3.4 & 3.5): 6 epochs Stage A Dual-VAE Zero-Shot domain alignment (`Recon L1+KL Loss: 0.0303`), followed by 10 epochs Stage B Frozen-VAE Latent Mapping fine-tuning across **3 random seeds (`42, 123, 999`)**
  * Loss Function: Compound `WanRepairLoss` (Normalized Hole $L_1$ + Valid $L_1$ + Multi-Scale Perceptual Edge/Gradient Loss + Latent Alignment $L_1$ + KL Divergence)
* **Quantitative Validation Metrics (vs. Classical OpenCV Baseline):**
  * Condition 1 (OpenCV Baseline — Telea + NLM + CLAHE): $\text{PSNR} = 23.78\text{ dB}$, $\text{SSIM} = 0.8502$, $\text{LPIPS} = 0.0973$
  * Condition 2 (Zero-Shot Wan Dual-VAE): $\text{PSNR} = 28.63\text{ dB}$, $\text{SSIM} = 0.9538$, $\text{LPIPS} = 0.0247$
  * Condition 3 (Fine-Tuned Wan Latent Mapping Net, 3 Seeds): **$\text{PSNR} = 39.29 \pm 0.08\text{ dB}$**, **$\text{SSIM} = 0.9842 \pm 0.0003$**, **$\text{LPIPS} = 0.0082 \pm 0.0001$**
* **Loading in Python / Pipeline Integration (Stage 4 & Stage 5):**
  ```python
  from src.repair import WanLatentRestorer

  restorer = WanLatentRestorer(checkpoint_path="models/repair_wan_latent_vae.pt", mode="fine_tuned")
  repaired_rgb = restorer.restore(damaged_rgb, damage_mask)  # Returns H x W x 3 uint8 RGB array
  ```
