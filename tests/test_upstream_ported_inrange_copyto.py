import numpy as np
import pytest

import cv2


pytestmark = pytest.mark.opencv_upstream


def test_upstream_like_copyto_with_mask_matches_numpy_copyto():
    # Reduced, data-free port of modules/python/test/test_copytomask.py.
    # Construct an image with a known "red-ish" region using BGR semantics.
    img = np.zeros((8, 9, 3), dtype=np.uint8)
    img[..., :] = [0, 0, 50]
    img[2:6, 3:7, :] = [10, 10, 200]  # within our range

    lower = np.array([0, 0, 100], dtype=np.uint8)
    upper = np.array([70, 70, 255], dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    assert mask.shape == img.shape[:2]

    dstcv = np.empty((img.shape[0] * 2, img.shape[1] * 2, 3), dtype=np.uint8)
    dstcv.fill(255)
    view = dstcv[: img.shape[0], : img.shape[1], :]
    cv2.copyTo(img, mask, view)

    dstnp = np.empty_like(dstcv)
    dstnp.fill(255)
    viewnp = dstnp[: img.shape[0], : img.shape[1], :]
    mask_bool = mask.astype(bool)
    _, mask_b = np.broadcast_arrays(img, mask_bool[..., None])
    np.copyto(viewnp, img, where=mask_b)

    assert cv2.norm(dstnp, dstcv, cv2.NORM_INF) == 0.0

