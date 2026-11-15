# notebooks

All model work happens in these Google Colab notebooks (T4 GPU). Each notebook's first cell clones this repository and installs `app/requirements.txt`.

| Notebook | What it does | Owner |
|---|---|---|
| `01_data_and_eda.ipynb` | Downloads the four datasets, explores them, preprocessing and the synthetic damage generator | Sai Krishna |
| `02_damage_detection.ipynb` | Fine-tunes the U-Net damage detector with 5-fold cross-validation | Sai Krishna |
| `03_repair.ipynb` | OpenCV baseline, zero-shot and fine-tuned repair runs (3 seeds) | Diaesh |
| `04_colorization_and_enhancement.ipynb` | DDColor, Real-ESRGAN and GFPGAN, with the skip rules for color input and photos without faces | Ashok |
| `05_pipeline_and_evaluation.ipynb` | Full pipeline, metrics and the ablation study on the real film scans | Diaesh, Ashok |

The notebook files are added as each phase starts.
