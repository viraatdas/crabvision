import numpy as np
import pytest

import cv2


pytestmark = pytest.mark.opencv_upstream


@pytest.mark.parametrize("norm_type", [cv2.NORM_INF, cv2.NORM_L1, cv2.NORM_L2])
@pytest.mark.parametrize("shape", [(1,), (7,), (3, 5), (4, 4), (2, 3, 3)])
@pytest.mark.parametrize(
    "dtype, gen",
    [
        (np.uint8, lambda rng, shape: rng.integers(0, 100, size=shape, dtype=np.uint8)),
        (np.int8, lambda rng, shape: rng.integers(-50, 50, size=shape, dtype=np.int8)),
        (np.uint16, lambda rng, shape: rng.integers(0, 1000, size=shape, dtype=np.uint16)),
        (np.int16, lambda rng, shape: rng.integers(-1000, 1000, size=shape, dtype=np.int16)),
        (np.int32, lambda rng, shape: rng.integers(-1000, 1000, size=shape, dtype=np.int32)),
        (np.float32, lambda rng, shape: rng.normal(0.0, 1.0, size=shape).astype(np.float32)),
        (np.float64, lambda rng, shape: rng.normal(0.0, 1.0, size=shape).astype(np.float64)),
    ],
)
def test_upstream_norm_matches_numpy(norm_type, shape, dtype, gen):
    # Reduced-but-broader port of upstream `modules/python/test/test_norm.py`.
    rng = np.random.default_rng(123)
    a = gen(rng, shape).astype(dtype, copy=False)
    a64 = a.astype(np.float64)

    if norm_type == cv2.NORM_INF:
        expected = np.linalg.norm(a64.ravel(), np.inf)
    elif norm_type == cv2.NORM_L1:
        expected = np.linalg.norm(a64.ravel(), 1)
    elif norm_type == cv2.NORM_L2:
        expected = np.linalg.norm(a64.ravel())
    else:
        raise AssertionError("unexpected norm type")

    actual = cv2.norm(a, norm_type)
    assert abs(expected - actual) < 1e-6


@pytest.mark.parametrize("norm_type", [cv2.NORM_INF, cv2.NORM_L1, cv2.NORM_L2])
@pytest.mark.parametrize("shape", [(7,), (3, 5), (2, 3, 3)])
@pytest.mark.parametrize(
    "dtype, gen",
    [
        (np.uint8, lambda rng, shape: rng.integers(0, 100, size=shape, dtype=np.uint8)),
        (np.int16, lambda rng, shape: rng.integers(-1000, 1000, size=shape, dtype=np.int16)),
        (np.int32, lambda rng, shape: rng.integers(-1000, 1000, size=shape, dtype=np.int32)),
        (np.float32, lambda rng, shape: rng.normal(0.0, 1.0, size=shape).astype(np.float32)),
        (np.float64, lambda rng, shape: rng.normal(0.0, 1.0, size=shape).astype(np.float64)),
    ],
)
def test_upstream_norm_two_arrays_matches_numpy(norm_type, shape, dtype, gen):
    rng = np.random.default_rng(456)
    a = gen(rng, shape).astype(dtype, copy=False)
    b = gen(rng, shape).astype(dtype, copy=False)
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
