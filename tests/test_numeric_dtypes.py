import numpy as np

import cv2


def test_add_subtract_absdiff_int16_saturating():
    a = np.array([30000, 30000, -30000], dtype=np.int16)
    b = np.array([10000, -10000, -10000], dtype=np.int16)

    add = cv2.add(a, b)
    sub = cv2.subtract(a, b)
    ad = cv2.absdiff(a, b)

    assert add.dtype == np.int16
    assert sub.dtype == np.int16
    assert ad.dtype == np.int16

    assert np.array_equal(add, np.array([32767, 20000, -32768], dtype=np.int16))
    assert np.array_equal(sub, np.array([20000, 32767, -20000], dtype=np.int16))
    assert np.array_equal(ad, np.array([20000, 32767, 20000], dtype=np.int16))


def test_add_subtract_absdiff_float32():
    a = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    b = np.array([2.0, 1.0, -0.25], dtype=np.float32)
    assert np.allclose(cv2.add(a, b), a + b)
    assert np.allclose(cv2.subtract(a, b), a - b)
    assert np.allclose(cv2.absdiff(a, b), np.abs(a - b))


def test_compare_float64_nan_behavior_matches_numpy():
    a = np.array([0.0, np.nan, 2.0], dtype=np.float64)
    b = np.array([0.0, np.nan, 1.0], dtype=np.float64)
    out = cv2.compare(a, b, cv2.CMP_EQ)
    expected = (a == b)
    assert np.array_equal(out != 0, expected)


def test_threshold_float32():
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    r, y = cv2.threshold(x, 1.0, 10.0, cv2.THRESH_BINARY)
    assert r == 1.0
    assert y.dtype == np.float32
    assert np.array_equal(y, np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32))

