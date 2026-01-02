import numpy as np

import cv2


def _replicate_index(i: int, n: int) -> int:
    return min(max(i, 0), n - 1)


def _separable_conv_i32(img: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img_c = img[:, :, None]
    else:
        img_c = img
    assert img_c.dtype == np.uint8
    h, w, c = img_c.shape
    rx = len(kx) // 2
    ry = len(ky) // 2

    tmp = np.zeros((h, w, c), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                s = 0
                for i, kw in enumerate(kx):
                    xx = _replicate_index(x + i - rx, w)
                    s += int(kw) * int(img_c[y, xx, ch])
                tmp[y, x, ch] = s

    out = np.zeros((h, w, c), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                s = 0
                for j, kw in enumerate(ky):
                    yy = _replicate_index(y + j - ry, h)
                    s += int(kw) * int(tmp[yy, x, ch])
                out[y, x, ch] = s

    return out[:, :, 0] if img.ndim == 2 else out


def test_sobel_dx_matches_reference_int16():
    img = np.zeros((7, 8), dtype=np.uint8)
    img[:, 4:] = 255
    deriv = np.array([-1, 0, 1], dtype=np.int32)
    smooth = np.array([1, 2, 1], dtype=np.int32)
    expected = _separable_conv_i32(img, deriv, smooth).astype(np.int16)
    out = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    assert out.dtype == np.int16
    assert np.array_equal(out, expected)


def test_scharr_dy_matches_reference_float32():
    img = np.zeros((9, 7), dtype=np.uint8)
    img[4:, :] = 200
    deriv = np.array([-1, 0, 1], dtype=np.int32)
    smooth = np.array([3, 10, 3], dtype=np.int32)
    expected_i32 = _separable_conv_i32(img, smooth, deriv)
    expected = expected_i32.astype(np.float32)
    out = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    assert out.dtype == np.float32
    assert np.array_equal(out, expected)


def test_sobel_supports_dst():
    img = np.zeros((5, 6), dtype=np.uint8)
    img[:, 3:] = 255
    dst = np.zeros_like(img, dtype=np.int16)
    out = cv2.Sobel(img, cv2.CV_16S, 1, 0, dst)
    assert out is not None
    assert np.array_equal(dst, out)

