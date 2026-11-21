import numpy as np
import pytest

from src.metrics import colorfulness, iou_f1, psnr


def test_psnr_identical_is_inf_and_known_value():
    a = np.full((8, 8, 3), 100, np.uint8)
    assert psnr(a, a) == float("inf")
    b = a + 1                                      # MSE = 1 -> 10*log10(255^2) = 48.13 dB
    assert psnr(b, a) == pytest.approx(48.13, abs=0.01)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4, 3), np.uint8), np.zeros((5, 4, 3), np.uint8))


def test_iou_f1():
    p = np.zeros((10, 10), np.uint8); p[:5] = 1
    t = np.zeros((10, 10), np.uint8); t[:5, :5] = 1
    r = iou_f1(p, t)
    assert r["iou"] == pytest.approx(0.5, abs=1e-3) and r["f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert iou_f1(np.zeros((4, 4)), np.zeros((4, 4))) == {"iou": 1.0, "f1": 1.0}


def test_colorfulness_gray_is_zero():
    g = np.full((16, 16, 3), 128, np.uint8)
    assert colorfulness(g) == 0.0
    c = g.copy(); c[..., 0] = 255
    assert colorfulness(c) > 10


def test_ssim_lpips_delta_e_run_when_installed():
    pytest.importorskip("skimage")
    from src.metrics import delta_e, ssim
    a = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert ssim(a, a) == pytest.approx(1.0) and delta_e(a, a) == pytest.approx(0.0)
