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
