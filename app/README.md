# app

Gradio demo for Hugging Face Spaces. This is the only folder with `.py` files: Spaces needs a Python entry point and cannot import code from a notebook.

| File | Contents |
|---|---|
| `app.py` | Gradio interface: upload a photo, choose the stages to run, compare input, mask and output, download the result |
| `pipeline.py` | `run_pipeline()` and the stage classes, exported from the notebooks so the notebooks and the app give the same result |
| `requirements.txt` | Pinned packages, shared by Spaces and the Colab notebooks |
