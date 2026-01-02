import numpy as np
import pytest

import cv2


pytestmark = pytest.mark.opencv_upstream


def test_upstream_like_bitwise_not_inplace():
    # Mirrors how upstream uses cv.bitwise_not(src, src) in modules/python/test/test_mser.py.
    src = np.array([[0, 1, 2], [10, 250, 255]], dtype=np.uint8)
    cv2.bitwise_not(src, src)
    assert np.array_equal(src, np.array([[255, 254, 253], [245, 5, 0]], dtype=np.uint8))


def test_upstream_like_bitwise_and_masked():
    # Upstream-style masked bitwise operations should only modify masked pixels.
    a = np.array([[0, 255], [170, 85]], dtype=np.uint8)
    b = np.array([[255, 255], [15, 240]], dtype=np.uint8)
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    dst = np.full_like(a, 7)
    cv2.bitwise_and(a, b, dst, mask)
    assert np.array_equal(dst, np.array([[7, 255], [10, 7]], dtype=np.uint8))


@pytest.mark.parametrize(
    "typ, expected",
    [
        (cv2.THRESH_BINARY, np.array([0, 0, 0, 255, 255], dtype=np.uint8)),
        (cv2.THRESH_BINARY_INV, np.array([255, 255, 255, 0, 0], dtype=np.uint8)),
        (cv2.THRESH_TRUNC, np.array([0, 1, 2, 2, 2], dtype=np.uint8)),
        (cv2.THRESH_TOZERO, np.array([0, 0, 0, 3, 255], dtype=np.uint8)),
        (cv2.THRESH_TOZERO_INV, np.array([0, 1, 2, 0, 0], dtype=np.uint8)),
    ],
)
def test_upstream_like_threshold_uint8(typ, expected):
    src = np.array([0, 1, 2, 3, 255], dtype=np.uint8)
    retval, dst = cv2.threshold(src, 2, 255, typ)
    assert retval == 2
    assert np.array_equal(dst, expected)


def test_upstream_like_count_non_zero():
    src = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    assert cv2.countNonZero(src) == 2
