# Old Photo Restoration Studio

> **Status:** work in progress. Folders are in place; code, notebooks and results are added phase by phase until the final submission on **Dec 7, 2026**.

A four-stage deep learning pipeline that restores old printed photos. You upload a scratched, dusty, creased or faded photo; the system finds the damage, repairs it, adds color to black-and-white or sepia photos, and sharpens the result, showing every stage side by side.

Built for **AAI-521 Applied Computer Vision for AI** (University of San Diego, MS Applied AI).

📄 Design document: [Old Photo Restoration Studio — Design Doc](https://docs.google.com/document/d/1BthOZSz6qmEfQ5B_y1QgVHVGNM1ZIqOb6d503z2xINA/edit?tab=t.0)

## Contents

- [Research question](#research-question)
- [How it works](#how-it-works)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Team](#team)
- [References](#references)

## Research question

Can pretrained deep models, fine-tuned on realistic synthetic damage, restore real old photos measurably better than classical image repair?

## How it works

A photo passes through four stages in order. Each stage can be switched off, so the ablation study measures what each one adds.

| Stage | Model | Input → output | Training Status |
|---|---|---|---|
| Preprocess | OpenCV | Raw upload → RGB image, tiles for large scans | Deterministic preprocessing |
| Detect damage | Scratch-detection U-Net (Wan et al., 2020) | Image → damage mask | Yes, trained (15 epochs, T4 GPU) |
| Repair | Learned tone correction (small CNN, trained) + LaMa inpainting (Suvorov et al., 2022, pretrained) on masks from an OpenPhoto-adapted copy of the U-Net. Baselines: OpenCV inpainting and Wan et al. (2020), pretrained and fine-tuned | Image (+ mask) → repaired image | Tone net trained (40 epochs x 3 seeds); detector copy fine-tuned (12 epochs); Wan et al. fine-tuned (5 epochs x 3 seeds); Colab L4 |
| Colorize | DDColor (Kang et al., 2023) | Gray or sepia image → color image | Pretrained |
| Enhance | Real-ESRGAN + GFPGAN (Wang et al., 2021) | Image → 4× upscaled image with restored faces | Pretrained |

A Gradio app on Hugging Face Spaces runs the same pipeline function as the notebooks: upload a photo, tick the stages to run, and compare input, mask and output.

Everything is designed to run on a free **Google Colab T4 (16 GB)**: 256 × 256 training crops, mixed precision, tiled inference and checkpoints saved to Google Drive every epoch.

## Datasets

| Dataset | Contents | Size | Used for |
|---|---|---|---|
| [OpenPhoto Restore](https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset) | 5,000 clean / damaged photo pairs | 4.4 GB | Training and testing the repair model (train 4,050 / val 450 / test 497, split by photo) |
| [humanface8000](https://www.kaggle.com/datasets/bharatadhikari/humanface8000) | 8,000 clean / damaged face pairs | 0.5 GB | Face repair and enhancement |
| [Real damaged film scans](https://doi.org/10.6084/m9.figshare.21803304) (Ivanova et al., 2023) | 44 real damaged 4K scans with expert restorations | 0.4 GB | Real-world test set |
| [ARTeFACT](https://huggingface.co/datasets/danielaivanova/damaged-media) (Ivanova et al., 2025) | 418 real damaged photos and prints with pixel damage masks | 4.1 GB | Training and testing the damage detector |

A synthetic damage generator adds scratches, dust, creases, torn corners, fading, sepia, grain and blur to clean images on the fly. Datasets are downloaded at run time into `data/` and are not committed.

## Evaluation

Metrics: PSNR, SSIM, LPIPS (repair quality), IoU and F1 (damage masks), ΔE CIEDE2000 (color), runtime per image.

| Run | Damage detection | Repair | Colorization | Enhancement |
|---|---|---|---|---|
| OpenCV baseline | OpenCV threshold | OpenCV inpaint | — | — |
| Pretrained repair | U-Net | Pretrained Wan et al., no extra training | — | — |
| Fine-tuned repair | U-Net | Fine-tuned Wan et al. | — | — |
| Full repair | OpenPhoto-adapted U-Net | Tone correction + LaMa | — | — |
| Repair + colorization | OpenPhoto-adapted U-Net | Tone correction + LaMa | DDColor | — |
| All stages | OpenPhoto-adapted U-Net | Tone correction + LaMa | DDColor | Real-ESRGAN + GFPGAN |

Damage detection uses 5-fold cross-validation on ARTeFACT; repair uses fixed held-out splits and 3 seeds; the whole pipeline is tested once, at the end, on the real film scans.

**Repair results** (497 held-out OpenPhoto test photos, luminance; `results/tables/s2_repair_ablation.csv`).

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | PSNR inside damage (dB) ↑ |
|---|---|---|---|---|
| Unrepaired input | 21.65 | 0.8551 | 0.1719 | 16.73 |
| Classical (OpenCV inpaint + denoise + contrast) | 20.12 | 0.8248 | 0.2253 | 16.00 |
| Pretrained Wan et al. (2020) | 21.76 | 0.8058 | 0.2092 | 17.06 |
| Fine-tuned Wan et al. (3 seeds) | 23.89 ± 0.05 | 0.8276 ± 0.0034 | 0.1983 ± 0.0030 | 20.60 ± 0.06 |
| Learned tone correction only (3 seeds) | 27.89 ± 0.08 | 0.8896 ± 0.0010 | 0.1546 ± 0.0009 | 18.70 ± 0.03 |
| **Full pipeline: adapted detector + tone correction + LaMa** | **28.73** | 0.8862 | **0.1325** | **21.56** |

The full pipeline improves on the input by +7.1 dB PSNR (paired bootstrap 95% CI +6.4 to +7.7; 86% of photos improve). Most of the gain comes from tone correction, because OpenPhoto damage is mostly fading and colour casts; LaMa inpainting then removes the scratches and tears (+2.9 dB inside the damage). The notebook-02 detector finds only 4% of OpenPhoto's damage, so the notebook fine-tunes a separate copy of it on OpenPhoto (recall 0.78, IoU 0.47 at threshold 0.7). Details: `notebooks/03_repair/03_repair.ipynb`.

## Repository structure

Model development and experiments run in Jupyter notebooks on Google Colab (T4 GPU). Reusable datasets, dataloaders, and helper modules are centralized in `src/` for cross-environment reproducibility.

```
old-photo-restoration-studio/
├── README.md
├── notebooks/                                # all model work, one Colab notebook per stage
│   ├── 01_data_and_preprocessing/            # datasets, EDA, preprocessing, synthetic damage generator
│   ├── 02_damage_detection/                  # U-Net damage detector, 15 epochs, T4 GPU
│   ├── 03_repair/                            # Baselines, Wan et al. fine-tuning, detector adaptation, tone correction + LaMa
│   ├── 04_colorization/                      # DDColor
│   ├── 05_enhancement/                       # Real-ESRGAN upscaling, GFPGAN faces
│   └── 06_pipeline_and_evaluation/           # full pipeline, metrics, ablation study
├── src/                                      # shared code imported by the notebooks
│   ├── damage_detection.py                   # U-Net damage detector + OpenCV detector
│   ├── repair.py                             # Classical repair, Wan et al. wrapper and fine-tuning
│   ├── tone.py                               # Learned tone-curve network
│   ├── inpaint.py                            # LaMa / Telea inpainting and the RepairPipeline
│   ├── detector_finetune.py                  # Fine-tunes a copy of the U-Net on OpenPhoto
│   ├── reference_masks.py                    # Reference damage masks from damaged/clean pairs
│   ├── metrics.py                            # PSNR/SSIM/LPIPS (luminance) and masked PSNR
│   └── data/                                 # DamageDataset, SyntheticTripletDataset, OpenPhoto splits
├── app/                                      # Gradio demo for Hugging Face Spaces
│   ├── app.py                                # Gradio interface
│   ├── pipeline.py                           # run_pipeline() and stage classes, exported from the notebooks
│   └── requirements.txt
├── tests/                                    # pytest checks for app/pipeline.py
├── docs/diagrams/                            # pipeline and timeline diagrams
├── data/                                     # datasets, downloaded at run time (not committed)
├── models/                                   # U-Net detectors, tone net, Wan et al. mapping shards
├── results/
│   ├── figures/                              # detection and repair verification figures
│   └── tables/                               # ablation and metric tables
├── report/                                   # final report (APA 7, PDF)
└── presentation/                             # slides and video
```

Each stage folder holds one notebook with the same name, for example `notebooks/03_repair/03_repair.ipynb`.

## Getting started

Setup instructions will be added as the code lands. The plan:

1. Open a stage notebook from `notebooks/` in Google Colab and choose a T4 GPU runtime.
2. Run all cells: the first cell clones this repository, installs `app/requirements.txt` and downloads the datasets.
3. Checkpoints are written to Google Drive so a disconnected session can resume.

## Team

| Member | Stages | Supporting components |
|---|---|---|
| N L N Sai Krishna Akula | Preprocessing, damage detection | Data loaders, synthetic damage generator, project report |
| Ashok Bhairwal | Colorization, enhancement | Gradio demo on Hugging Face Spaces, evaluation harness and metrics |
| Diaesh Antony | Repair and the OpenCV baseline | Pipeline orchestration, README, ablation runs, presentation slides, video |

## References

- Wan, Z., et al. (2020). Bringing old photos back to life. *CVPR 2020*.
- Wan, Z., et al. (2022). Old photo restoration via deep latent space translation. *IEEE TPAMI*. https://doi.org/10.1109/tpami.2022.3163183
- Kang, X., et al. (2023). DDColor: Towards photo-realistic image colorization via dual decoders. *ICCV 2023*. https://doi.org/10.1109/iccv51070.2023.00037
- Wang, X., et al. (2021). Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. *ICCVW 2021*. https://doi.org/10.1109/iccvw54120.2021.00217
- Wang, X., et al. (2021). Towards real-world blind face restoration with generative facial prior. *CVPR 2021*. https://doi.org/10.1109/cvpr46437.2021.00905
- Ivanova, D., et al. (2025). ARTeFACT: Benchmarking segmentation models on diverse analogue media damage. *WACV 2025*. https://doi.org/10.1109/wacv61041.2025.00723
- Ivanova, D., et al. (2023). Simulating analogue film damage to analyse and improve artefact restoration on high-resolution scans. *Computer Graphics Forum*. https://doi.org/10.1111/cgf.14749
- Suvorov, R., et al. (2022). Resolution-robust large mask inpainting with Fourier convolutions. *WACV 2022*. https://doi.org/10.1109/wacv51458.2022.00323
- Telea, A. (2004). An image inpainting technique based on the fast marching method. *Journal of Graphics Tools*, 9(1).

The full design, evaluation plan and corner cases are in the design document.
