# notebooks

All model work happens in Google Colab notebooks (T4 GPU), one folder per pipeline stage. Each notebook's first cell clones this repository and installs `app/requirements.txt`.

| Folder | Notebook | Owner |
|---|---|---|
| `01_data_and_preprocessing/` | `01_data_and_preprocessing.ipynb` | Sai Krishna |
| `02_damage_detection/` | `02_damage_detection.ipynb` | Sai Krishna |
| `03_repair/` | `03_repair.ipynb` | Diaesh |
| `04_colorization/` | `04_colorization.ipynb` | Ashok |
| `05_enhancement/` | `05_enhancement.ipynb` | Ashok |
| `06_pipeline_and_evaluation/` | `06_pipeline_and_evaluation.ipynb` | Diaesh and Ashok |

Each stage folder also holds that stage's small outputs (sample images, plots) that the report needs.
