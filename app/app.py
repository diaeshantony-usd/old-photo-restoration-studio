"""Gradio demo: upload an old photo, choose the stages, compare every step, download the result.

Run locally or on Hugging Face Spaces:  python app/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ALL_STAGES, get_default_pipeline, prepare_image  # noqa: E402

NOTICE = ("Output is AI-restored. Colors are a plausible guess, not the true colors, and restored faces "
          "may differ from the real person. Compare with the input before relying on the result.")


def restore(photo: Image.Image, stages: list, face_weight: float):
    if photo is None:
        raise gr.Error("Upload a photo first.")
    if not stages:
        raise gr.Error("Tick at least one stage.")
    image = prepare_image(photo)
    ordered = [s for s in ALL_STAGES if s in stages]
    result = get_default_pipeline().run(image, ordered, face_weight=face_weight)

    gallery = [(result["input"], "Input")]
    if result["mask"] is not None:
        gallery.append(((result["mask"] > 0).astype(np.uint8) * 255, "Damage mask"))
    for key, label, stage in (("repaired", "After repair", "repair"), ("colorized", "After colorization", "colorize")):
        if stage in ordered:
            gallery.append((result[key], label))
    gallery.append((result["output"], "Final output"))

    path = Path(tempfile.mkdtemp()) / "restored.png"
    Image.fromarray(result["output"]).save(path)
    return gallery, "\n".join(result["log"]), str(path)


with gr.Blocks(title="Old Photo Restoration Studio") as demo:
    gr.Markdown("# Old Photo Restoration Studio\nDetect damage, repair, colorize and sharpen an old photo. "
                "Every stage is shown side by side.")
    with gr.Row():
        with gr.Column():
            photo = gr.Image(type="pil", label="Old photo")
            stages = gr.CheckboxGroup(ALL_STAGES, value=list(ALL_STAGES), label="Stages to run")
            face_weight = gr.Slider(0.0, 1.0, value=0.7, step=0.1,
                                    label="Face restoration strength (lower keeps more of the original face)")
            go = gr.Button("Restore", variant="primary")
        with gr.Column():
            gallery = gr.Gallery(label="Stage by stage", columns=2, height=520)
            log = gr.Textbox(label="What ran", lines=6)
            download = gr.File(label="Download result")
    gr.Markdown(NOTICE)
    go.click(restore, [photo, stages, face_weight], [gallery, log, download])

if __name__ == "__main__":
    demo.launch()
