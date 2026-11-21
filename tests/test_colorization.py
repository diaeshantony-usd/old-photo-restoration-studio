"""Skip-rule tests that need no model. Model-based checks run in notebooks/04_colorization."""
import numpy as np
import pytest

from src.colorization import Colorizer, classify_color, is_grayscale, to_gray, to_sepia


@pytest.fixture(scope="module")
def color_image():
    rng = np.random.default_rng(0)
    import cv2
    im = cv2.GaussianBlur(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), (0, 0), 12)
    return cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)


class FakePipeline:
    input_size = 512
    calls = 0

    def process(self, bgr):
        self.calls += 1
        out = bgr.copy()
        out[..., 2] = 200          # add red so the output visibly has color
        return out


def test_classes(color_image):
    assert classify_color(color_image) == "color"
    assert classify_color(to_gray(color_image)) == "gray"
    assert classify_color(to_sepia(to_gray(color_image))) == "sepia"
    assert is_grayscale(to_gray(color_image)) and not is_grayscale(color_image)


def test_color_input_skips_colorization(color_image):
    fake = FakePipeline()
    c = Colorizer(fake)
    out = c.colorize(color_image)
    assert fake.calls == 0 and not c.last_info["ran"] and np.array_equal(out, color_image)


def test_gray_input_is_colorized(color_image):
    fake = FakePipeline()
    c = Colorizer(fake)
    src = to_gray(color_image)
    out = c.colorize(src)
    assert fake.calls == 1 and c.last_info["ran"] and out.shape == src.shape


def test_switched_off_returns_input(color_image):
    c = Colorizer(FakePipeline(), enabled=False)
    src = to_gray(color_image)
    assert np.array_equal(c.colorize(src), src) and not c.last_info["ran"]
