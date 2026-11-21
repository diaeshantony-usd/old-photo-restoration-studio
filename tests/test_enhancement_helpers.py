"""Helper-logic tests that need no model. Model-based checks run in notebooks/05_enhancement."""
from src.enhancement import MAX_OUTPUT_SIDE, compute_outscale, keep_face


def test_output_cap():
    assert compute_outscale(1000, 1000, 4.0) == 4.0
    assert compute_outscale(3000, 4000, 4.0) == 1.024
    assert compute_outscale(8000, 6000, 4.0) == 1.0
    assert max(3000, 4000) * compute_outscale(3000, 4000) <= MAX_OUTPUT_SIDE


def test_face_size_filter():
    assert keep_face([0, 0, 40, 50, 0.9])
    assert not keep_face([0, 0, 20, 50, 0.9])
    assert not keep_face([0, 0, 50, 31, 0.9])
