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
def test_upstream_lena_imread_imdecode_consistency():
    # Equivalent of OpenCV's tests_common.get_sample logic, but for our Rust-backed cv2.
    # Uses opencv_extra test data: testdata/cv/shared/lena.png
    data_root = _opencv_extra_path()
    assert data_root is not None

    lena = data_root / "cv" / "shared" / "lena.png"
    if not lena.exists():
        pytest.skip(f"missing file: {lena}")

    img_from_path = cv2.imread(lena, cv2.IMREAD_COLOR)
    assert img_from_path is not None
    assert img_from_path.ndim == 3
    assert img_from_path.shape[2] == 3

    raw = lena.read_bytes()
    buf = np.frombuffer(raw, dtype=np.uint8)
    img_from_buf = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert img_from_buf is not None
    assert np.array_equal(img_from_buf, img_from_path)
