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

| Stage | Model | Input → output | Trained by us? |
|---|---|---|---|
| Preprocess | OpenCV | Raw upload → RGB image, tiles for large scans | No |
| Detect damage | Scratch-detection U-Net (Wan et al., 2020) | Image → damage mask | Yes, fine-tuned |
| Repair | Latent-space restoration (Wan et al., 2020); OpenCV inpainting as baseline | Image + mask → repaired image | Yes, fine-tuned |
| Colorize | DDColor (Kang et al., 2023) | Gray or sepia image → color image | No, pretrained |
| Enhance | Real-ESRGAN + GFPGAN (Wang et al., 2021) | Image → 4× upscaled image with restored faces | No, pretrained |

A Gradio app on Hugging Face Spaces runs the same pipeline function as the notebooks: upload a photo, tick the stages to run, and compare input, mask and output.

Everything is designed to run on a free **Google Colab T4 (16 GB)**: 256 × 256 training crops, mixed precision, tiled inference and checkpoints saved to Google Drive every epoch.

## Datasets

| Dataset | Contents | Size | Used for |
|---|---|---|---|
| [OpenPhoto Restore](https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset) | 5,000 clean / damaged photo pairs | 4.4 GB | Training the repair model |
| [humanface8000](https://www.kaggle.com/datasets/bharatadhikari/humanface8000) | 8,000 clean / damaged face pairs | 0.5 GB | Face repair and enhancement |
| [Real damaged film scans](https://doi.org/10.6084/m9.figshare.21803304) (Ivanova et al., 2023) | 44 real damaged 4K scans with expert restorations | 0.4 GB | Real-world test set |
| [ARTeFACT](https://huggingface.co/datasets/danielaivanova/damaged-media) (Ivanova et al., 2025) | 418 real damaged photos and prints with pixel damage masks | 4.1 GB | Training and testing the damage detector |

A synthetic damage generator adds scratches, dust, creases, torn corners, fading, sepia, grain and blur to clean images on the fly. Datasets are downloaded at run time into `data/` and are not committed.

## Evaluation

Metrics: PSNR, SSIM, LPIPS (repair quality), IoU and F1 (damage masks), ΔE CIEDE2000 (color), runtime per image.

| Run | Damage detection | Repair | Colorization | Enhancement |
|---|---|---|---|---|
| OpenCV baseline | OpenCV threshold | OpenCV inpaint | — | — |
| Zero-shot repair | U-Net | Pretrained, no extra training | — | — |
| Fine-tuned repair | U-Net | Fine-tuned | — | — |
| Repair + colorization | U-Net | Fine-tuned | DDColor | — |
| All stages | U-Net | Fine-tuned | DDColor | Real-ESRGAN + GFPGAN |

Damage detection uses 5-fold cross-validation on ARTeFACT; repair uses fixed held-out splits and 3 seeds; the whole pipeline is tested once, at the end, on the real film scans.

## Repository structure

All model work is done in Jupyter notebooks that run in Google Colab. The only Python files are in `app/`, because the Hugging Face Spaces demo needs a `.py` entry point and cannot import code from a notebook.

```
old-photo-restoration-studio/
├── README.md
├── notebooks/                                # all model work, as Colab notebooks
│   ├── 01_data_and_eda.ipynb                 # datasets, EDA, synthetic damage generator
│   ├── 02_damage_detection.ipynb             # U-Net damage detector, 5-fold cross-validation
│   ├── 03_repair.ipynb                       # OpenCV baseline, zero-shot and fine-tuned repair
│   ├── 04_colorization_and_enhancement.ipynb # DDColor, Real-ESRGAN, GFPGAN
│   └── 05_pipeline_and_evaluation.ipynb      # full pipeline, metrics, ablation study
├── app/                                      # Gradio demo for Hugging Face Spaces (the only .py files)
│   ├── app.py                                # Gradio interface
│   ├── pipeline.py                           # run_pipeline() and stage classes, exported from the notebooks
│   └── requirements.txt
├── tests/                                    # pytest checks for app/pipeline.py
├── docs/diagrams/                            # pipeline and timeline diagrams
├── data/                                     # datasets, downloaded at run time (not committed)
├── models/                                   # checkpoints (not committed; saved to Google Drive)
├── results/
│   ├── figures/                              # before / mask / after grids, training curves
│   └── tables/                               # ablation and metric tables
├── report/                                   # final report (APA 7, PDF)
└── presentation/                             # slides and video
```

The notebook files are added as each phase starts.

## Getting started

Setup instructions will be added as the code lands. The plan:

1. Open a notebook from `notebooks/` in Google Colab and choose a T4 GPU runtime.
2. Run all cells: the first cell clones this repository, installs `app/requirements.txt` and downloads the datasets.
3. Checkpoints are written to Google Drive so a disconnected session can resume.

## Team

| Member | Stages | Supporting components |
|---|---|---|
| N L N Sai Krishna Akula | Preprocessing, damage detection | Data loaders, synthetic damage generator, project report |
| Ashok Bhairwal | Colorization, enhancement | Gradio demo on Hugging Face Spaces, evaluation harness and metrics |
| Diaesh Antony | Repair and the OpenCV baseline | Pipeline orchestration, README, ablation runs, presentation slides, video |

**Workflow:** work on a feature branch, open a pull request for a teammate to review, run `ruff check .` and `pytest` before pushing, and never commit datasets, checkpoints or API keys.

## References

- Wan, Z., et al. (2020). Bringing old photos back to life. *CVPR 2020*.
- Wan, Z., et al. (2022). Old photo restoration via deep latent space translation. *IEEE TPAMI*. https://doi.org/10.1109/tpami.2022.3163183
- Kang, X., et al. (2023). DDColor: Towards photo-realistic image colorization via dual decoders. *ICCV 2023*. https://doi.org/10.1109/iccv51070.2023.00037
- Wang, X., et al. (2021). Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. *ICCVW 2021*. https://doi.org/10.1109/iccvw54120.2021.00217
- Wang, X., et al. (2021). Towards real-world blind face restoration with generative facial prior. *CVPR 2021*. https://doi.org/10.1109/cvpr46437.2021.00905
- Ivanova, D., et al. (2025). ARTeFACT: Benchmarking segmentation models on diverse analogue media damage. *WACV 2025*. https://doi.org/10.1109/wacv61041.2025.00723
- Ivanova, D., et al. (2023). Simulating analogue film damage to analyse and improve artefact restoration on high-resolution scans. *Computer Graphics Forum*. https://doi.org/10.1111/cgf.14749
- Telea, A. (2004). An image inpainting technique based on the fast marching method. *Journal of Graphics Tools*, 9(1).

The full design, evaluation plan and corner cases are in the design document.
