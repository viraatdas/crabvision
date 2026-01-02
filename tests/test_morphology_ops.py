import numpy as np

import cv2


def test_get_structuring_element_shapes():
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    assert k.shape == (5, 3)
    assert set(np.unique(k)).issubset({0, 1})
    assert int(np.sum(k)) == 15

    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    assert int(np.sum(k)) == 5

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    assert k.shape == (5, 5)
    assert 0 < int(np.sum(k)) < 25


def test_erode_dilate_ordering_grayscale():
    img = np.zeros((7, 7), dtype=np.uint8)
    img[3, 3] = 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    e = cv2.erode(img, k)
    d = cv2.dilate(img, k)

    assert int(np.max(e)) <= int(np.max(img))
    assert int(np.max(d)) >= int(np.max(img))
    assert int(np.min(e)) >= 0


def test_morphology_ex_gradient_matches_dilate_minus_erode():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(16, 17), dtype=np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    g = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)
    d = cv2.dilate(img, k)
    e = cv2.erode(img, k)
    assert np.array_equal(g, cv2.subtract(d, e))


def test_open_close_idempotent_on_constant_image():
    img = np.full((9, 8, 3), 123, dtype=np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    assert np.array_equal(opened, img)
    assert np.array_equal(closed, img)

