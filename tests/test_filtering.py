import numpy as np

import cv2


def _replicate_index(i: int, n: int) -> int:
    return min(max(i, 0), n - 1)


def _separable_conv_u8(img: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    assert img.dtype == np.uint8
    if img.ndim == 2:
        img_c = img[:, :, None]
    else:
        img_c = img

    h, w, c = img_c.shape
    rx = len(kx) // 2
    ry = len(ky) // 2

    tmp = np.zeros((h, w, c), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                s = 0.0
                for i, kw in enumerate(kx):
                    xx = _replicate_index(x + i - rx, w)
                    s += float(kw) * float(img_c[y, xx, ch])
                tmp[y, x, ch] = s

    out = np.zeros((h, w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                s = 0.0
                for j, kw in enumerate(ky):
                    yy = _replicate_index(y + j - ry, h)
                    s += float(kw) * float(tmp[yy, x, ch])
                out[y, x, ch] = np.uint8(np.clip(np.round(s), 0, 255))

    return out[:, :, 0] if img.ndim == 2 else out


def _gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    assert ksize > 0 and ksize % 2 == 1
    if sigma == 0.0:
        r = (ksize - 1) * 0.5
        sigma = 0.3 * (r - 1.0) + 0.8
    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k.astype(np.float32)


def test_blur_constant_image_is_constant():
    img = np.full((9, 7, 3), 123, dtype=np.uint8)
    out = cv2.blur(img, (5, 3))
    assert np.array_equal(out, img)


def test_gaussian_blur_matches_reference_replication_border():
    img = np.zeros((7, 8), dtype=np.uint8)
    img[3, 4] = 255
    kx = _gaussian_kernel_1d(5, 1.2)
    ky = _gaussian_kernel_1d(3, 0.8)

    expected = _separable_conv_u8(img, kx, ky)
    out = cv2.GaussianBlur(img, (5, 3), 1.2, sigmaY=0.8)
    assert np.array_equal(out, expected)


def test_gaussian_blur_supports_dst():
    img = np.zeros((5, 6), dtype=np.uint8)
    img[2, 3] = 255
    dst = np.empty_like(img)
    out = cv2.GaussianBlur(img, (3, 3), 1.0, dst)
    assert out is not None
    assert np.array_equal(dst, out)

