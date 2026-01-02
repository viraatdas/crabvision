import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

import cv2


pytestmark = pytest.mark.opencv_upstream


def _opencv_extra_path() -> Optional[Path]:
    root = os.environ.get("OPENCV_TEST_DATA_PATH")
    if not root:
        return None
    return Path(root)


@pytest.mark.skipif(
    _opencv_extra_path() is None,
    reason="set OPENCV_TEST_DATA_PATH (see scripts/sync_upstream_tests.sh)",
)
def test_upstream_imread_to_buffer_like_opencv():
    # Mirrors OpenCV upstream test `modules/python/test/test_imread.py::test_imread_to_buffer`.
    data_root = _opencv_extra_path()
    assert data_root is not None

    path = data_root / "cv" / "shared" / "lena.png"
    if not path.exists():
        pytest.skip(f"missing file: {path}")

    ref = cv2.imread(path)
    assert ref is not None

    dst = np.zeros_like(ref)
    out = cv2.imread(path, dst)
    assert out is not None

    # dst is filled in-place.
    assert cv2.norm(ref, dst, cv2.NORM_INF) == 0.0

