import numpy as np

import cv2


def test_flip_codes_match_expectations_grayscale():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)

    v = cv2.flip(img, 0)
    assert np.array_equal(v, img[::-1, :])

    h = cv2.flip(img, 1)
    assert np.array_equal(h, img[:, ::-1])

    b = cv2.flip(img, -1)
    assert np.array_equal(b, img[::-1, ::-1])


def test_transpose_roundtrip_color():
    img = np.zeros((5, 7, 3), dtype=np.uint8)
    img[..., 0] = np.arange(5, dtype=np.uint8)[:, None]
    img[..., 1] = 10
    img[..., 2] = 200

    t = cv2.transpose(img)
    assert t.shape == (7, 5, 3)
    back = cv2.transpose(t)
    assert np.array_equal(back, img)


def test_rotate_codes_match_numpy():
    img = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
    r90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    assert np.array_equal(r90, np.rot90(img, k=3))
    r180 = cv2.rotate(img, cv2.ROTATE_180)
    assert np.array_equal(r180, np.rot90(img, k=2))
    r270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert np.array_equal(r270, np.rot90(img, k=1))

