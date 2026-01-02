import os
import io
import tempfile

import numpy as np
import cv2


def test_cvtcolor_bgr2rgb_roundtrip():
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[0, 0] = [10, 20, 30]  # B,G,R
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert rgb.shape == (2, 3, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert np.array_equal(img, bgr)


def test_cvtcolor_bgr2gray_and_back():
    img = np.zeros((3, 2, 3), dtype=np.uint8)
    img[..., :] = [50, 100, 150]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (3, 2)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert bgr.shape == (3, 2, 3)
    # All channels equal after expanding gray
    assert np.all(bgr[..., 0] == bgr[..., 1]) and np.all(bgr[..., 1] == bgr[..., 2])


def test_resize_nearest_and_linear():
    src = np.arange(0, 4 * 4, dtype=np.uint8).reshape(4, 4)
    dst_nearest = cv2.resize(src, (2, 2), interpolation=cv2.INTER_NEAREST)
    dst_linear = cv2.resize(src, (2, 2), interpolation=cv2.INTER_LINEAR)
    assert dst_nearest.shape == (2, 2)
    assert dst_linear.shape == (2, 2)


def test_imwrite_and_imread_color():
    img = np.zeros((8, 9, 3), dtype=np.uint8)
    img[0, 0] = [0, 0, 255]  # Red in BGR
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.png")
        assert cv2.imwrite(path, img)
        out = cv2.imread(path, cv2.IMREAD_COLOR)
        assert out.shape == (8, 9, 3)
        # Expect the pixel we wrote to be red in BGR space
        assert np.array_equal(out[0, 0], [0, 0, 255])


def test_imwrite_and_imread_gray():
    img = np.full((7, 5), 127, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.jpg")
        assert cv2.imwrite(path, img)
        out = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert out.shape == (7, 5)
        assert out.dtype == np.uint8


def test_imread_missing_returns_none():
    out = cv2.imread("/this/path/should/not/exist.png")
    assert out is None


def test_imencode_imdecode_png_roundtrip():
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    img[0, 0] = [1, 2, 3]  # BGR
    ok, buf = cv2.imencode(".png", img)
    assert ok is True
    assert buf.dtype == np.uint8
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == img.shape
    assert np.array_equal(decoded, img)


def test_norm_and_absdiff_basics():
    a = np.array([0, 10, 200], dtype=np.uint8)
    b = np.array([0, 20, 100], dtype=np.uint8)
    d = cv2.absdiff(a, b)
    assert np.array_equal(d, np.array([0, 10, 100], dtype=np.uint8))
    assert cv2.norm(a, b, cv2.NORM_INF) == 100.0


def test_add_and_subtract_saturate_uint8():
    a = np.array([250, 10], dtype=np.uint8)
    b = np.array([10, 50], dtype=np.uint8)
    assert np.array_equal(cv2.add(a, b), np.array([255, 60], dtype=np.uint8))
    assert np.array_equal(cv2.subtract(a, b), np.array([240, 0], dtype=np.uint8))


def test_split_merge_roundtrip():
    img = np.zeros((3, 4, 3), dtype=np.uint8)
    img[..., 0] = 1
    img[..., 1] = 2
    img[..., 2] = 3
    ch = cv2.split(img)
    assert len(ch) == 3
    merged = cv2.merge(ch)
    assert np.array_equal(merged, img)


def test_bitwise_ops_and_compare():
    a = np.array([0, 1, 2, 255], dtype=np.uint8)
    b = np.array([0, 2, 2, 1], dtype=np.uint8)

    assert np.array_equal(cv2.bitwise_not(a), np.array([255, 254, 253, 0], dtype=np.uint8))
    assert np.array_equal(cv2.bitwise_and(a, b), np.array([0, 0, 2, 1], dtype=np.uint8))
    assert np.array_equal(cv2.bitwise_or(a, b), np.array([0, 3, 2, 255], dtype=np.uint8))
    assert np.array_equal(cv2.bitwise_xor(a, b), np.array([0, 3, 0, 254], dtype=np.uint8))

    eq = cv2.compare(a, b, cv2.CMP_EQ)
    assert np.array_equal(eq, np.array([255, 0, 255, 0], dtype=np.uint8))


def test_bitwise_and_with_mask_and_dst():
    a = np.array([[0, 255], [170, 85]], dtype=np.uint8)
    b = np.array([[255, 255], [15, 240]], dtype=np.uint8)
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    dst = np.full_like(a, 7)
    out = cv2.bitwise_and(a, b, dst, mask)
    assert out is not None
    # Where mask==0, dst stays 7; else it's a&b.
    expected = np.array([[7, 255], [10, 7]], dtype=np.uint8)
    assert np.array_equal(dst, expected)


def test_inrange_and_copyto_masked_copy():
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    img[..., :] = [5, 5, 5]
    img[1:3, 1:4, :] = [10, 10, 200]

    mask = cv2.inRange(img, [0, 0, 100], [70, 70, 255])
    assert mask.shape == (4, 5)

    dst = np.full_like(img, 255)
    out = cv2.copyTo(img, mask, dst)
    assert out is not None

    # Where mask is 0, dst stays 255; where mask is 255, dst equals src.
    mask_b = mask.astype(bool)[..., None]
    expected = np.where(mask_b, img, 255)
    assert np.array_equal(dst, expected)
