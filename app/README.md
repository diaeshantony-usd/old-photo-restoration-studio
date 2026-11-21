# app

Gradio demo for Hugging Face Spaces: upload a photo, choose the stages to run, compare every step, download the result.

| File | Contents |
|---|---|
| `app.py` | Gradio interface: upload, stage checkboxes, face-restoration strength slider, stage-by-stage gallery, log of what ran, download button |
| `pipeline.py` | `Pipeline.run()` / `run_pipeline()`: runs detect, repair, colorize, enhance in order and returns every intermediate image plus a log. Also `prepare_image()` (EXIF rotation, alpha flattened on white, RGB) |
| `requirements.txt` | Packages shared by Spaces and the Colab notebooks (lower bounds for now; freeze after the first full Colab run) |

The stage code is in `src/`: `src/colorization.py` and `src/enhancement.py` (Ashok), `src/metrics.py` (shared). The detector and repair model plug in as objects with `predict(image) -> mask` and `restore(image, mask) -> image`. Until they are connected, those stages are skipped and the log says so.

Run locally: `python app/app.py`.
