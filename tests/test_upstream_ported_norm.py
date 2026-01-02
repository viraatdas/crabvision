import numpy as np
import pytest

import cv2


pytestmark = pytest.mark.opencv_upstream


@pytest.mark.parametrize("norm_type", [cv2.NORM_INF, cv2.NORM_L1, cv2.NORM_L2])
@pytest.mark.parametrize("shape", [(1,), (7,), (3, 5), (4, 4), (2, 3, 3)])
def test_upstream_norm_matches_numpy_for_uint8(norm_type, shape):
    # A reduced port of OpenCV upstream `modules/python/test/test_norm.py`.
    # We only support uint8 currently.
    rng = np.random.default_rng(123)
    a = rng.integers(0, 100, size=shape, dtype=np.uint8)

    if norm_type == cv2.NORM_INF:
        expected = np.linalg.norm(a.astype(np.float64).ravel(), np.inf)
    elif norm_type == cv2.NORM_L1:
        expected = np.linalg.norm(a.astype(np.float64).ravel(), 1)
    elif norm_type == cv2.NORM_L2:
        expected = np.linalg.norm(a.astype(np.float64).ravel())
    else:
        raise AssertionError("unexpected norm type")

    actual = cv2.norm(a, norm_type)
    assert abs(expected - actual) < 1e-6


@pytest.mark.parametrize("norm_type", [cv2.NORM_INF, cv2.NORM_L1, cv2.NORM_L2])
@pytest.mark.parametrize("shape", [(7,), (3, 5), (2, 3, 3)])
def test_upstream_norm_two_arrays_matches_numpy_for_uint8(norm_type, shape):
    rng = np.random.default_rng(456)
    a = rng.integers(0, 100, size=shape, dtype=np.uint8)
    b = rng.integers(0, 100, size=shape, dtype=np.uint8)
    diff = a.astype(np.float64) - b.astype(np.float64)

    if norm_type == cv2.NORM_INF:
        expected = np.linalg.norm(diff.ravel(), np.inf)
    elif norm_type == cv2.NORM_L1:
        expected = np.linalg.norm(diff.ravel(), 1)
    elif norm_type == cv2.NORM_L2:
        expected = np.linalg.norm(diff.ravel())
    else:
        raise AssertionError("unexpected norm type")

    actual = cv2.norm(a, b, norm_type)
    assert abs(expected - actual) < 1e-6


def test_upstream_norm_rejects_scalar_as_src2():
    with pytest.raises(Exception):
        cv2.norm(np.array([1, 2], dtype=np.uint8), 123, cv2.NORM_L2)

