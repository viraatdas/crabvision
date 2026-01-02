import numpy as np

import cv2


def test_canny_step_edge_detects_single_vertical_edge_line():
    h, w = 12, 16
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, w // 2 :] = 255
    edges = cv2.Canny(img, 50, 100)

    assert edges.shape == img.shape
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)).issubset({0, 255})

    # For a sharp vertical step, edges should form a near-vertical 1px line.
    # We ignore the outer border because our implementation zeroes borders.
    inner = edges[1:-1, 1:-1]
    cols = np.where(inner == 255)[1]
    assert cols.size > 0
    # Expect the edge to be concentrated around the boundary column.
    boundary = (w // 2) - 1
    assert np.max(np.abs(cols - boundary)) <= 1


def test_canny_threshold_monotonicity():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    e1 = cv2.Canny(img, 10, 20)
    e2 = cv2.Canny(img, 50, 100)
    assert int(np.count_nonzero(e2)) <= int(np.count_nonzero(e1))


def test_canny_supports_edges_dst():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:, 5:] = 255
    dst = np.zeros_like(img)
    out = cv2.Canny(img, 50, 100, dst)
    assert out is not None
    assert np.array_equal(dst, out)


def test_canny_accepts_color_input_like_opencv():
    rng = np.random.default_rng(1)
    color = rng.integers(0, 256, size=(32, 31, 3), dtype=np.uint8)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    e1 = cv2.Canny(color, 30, 80)
    e2 = cv2.Canny(gray, 30, 80)
    assert np.array_equal(e1, e2)
