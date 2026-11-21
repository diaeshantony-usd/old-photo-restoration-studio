import numpy as np
from PIL import Image

from pipeline import Pipeline, prepare_image


class FakeColorizer:
    last_info = {}

    def colorize(self, im):
        self.last_info = dict(ran=True, kind="gray", reason="ok")
        return im


class FakeEnhancer:
    last_info = {}
    face_weight = 0.7

    def enhance(self, im):
        self.last_info = dict(ran=True, outscale=2.0, face_step="skipped (no face found)")
        return np.kron(im, np.ones((2, 2, 1), np.uint8))


def test_missing_stages_are_skipped_and_logged():
    im = np.zeros((16, 16, 3), np.uint8)
    out = Pipeline().run(im)
    assert np.array_equal(out["output"], im)
    assert sum("not available yet" in l for l in out["log"]) == 4


def test_stage_order_and_switches():
    im = np.zeros((16, 16, 3), np.uint8)
    p = Pipeline(colorizer=FakeColorizer(), enhancer=FakeEnhancer())
    assert p.run(im, ["colorize"])["output"].shape == im.shape            # enhance off
    assert p.run(im, ["colorize", "enhance"])["output"].shape == (32, 32, 3)
    assert p.run(im, [])["output"] is im                                    # everything off


def test_face_weight_is_forwarded():
    p = Pipeline(enhancer=FakeEnhancer())
    p.run(np.zeros((8, 8, 3), np.uint8), ["enhance"], face_weight=0.3)
    assert p.enhancer.face_weight == 0.3


def test_prepare_image_flattens_alpha_and_returns_rgb():
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))                      # fully transparent
    arr = prepare_image(rgba)
    assert arr.shape == (10, 10, 3) and arr.min() == 255                   # flattened onto white
