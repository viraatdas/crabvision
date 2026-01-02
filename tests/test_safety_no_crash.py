import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )


def test_no_crash_on_wrong_dtypes_and_shapes():
    # This test is specifically designed to catch "hard crash" regressions.
    # If the extension segfaults, the subprocess exits non-zero.
    code = r"""
import numpy as np
import cv2

def must_raise(fn):
    try:
        fn()
    except Exception:
        return
    raise AssertionError("expected exception")

must_raise(lambda: cv2.cvtColor(np.zeros((2,2), dtype=np.float32), cv2.COLOR_BGR2GRAY))
must_raise(lambda: cv2.resize(np.zeros((2,2,4), dtype=np.uint8), (3,3)))
must_raise(lambda: cv2.merge([np.zeros((2,2), np.uint8), np.zeros((2,3), np.uint8), np.zeros((2,2), np.uint8)]))
must_raise(lambda: cv2.add(np.zeros((2,2), np.uint8), np.zeros((2,3), np.uint8)))

# imread should fail gracefully (return None) even if a dst buffer is provided.
dst = np.zeros((2,2,3), np.uint8)
out = cv2.imread("/definitely/missing.png", dst)
assert out is None
assert np.all(dst == 0)

print("ok")
"""
    p = _run(code)
    assert p.returncode == 0, p.stderr
    assert "ok" in p.stdout


def test_no_crash_on_non_contiguous_inputs():
    code = r"""
import numpy as np
import cv2

img = np.zeros((10, 12, 3), dtype=np.uint8)
img[..., 0] = np.arange(10, dtype=np.uint8)[:, None]

view = img[::2, ::3, :]
ch = cv2.split(view)
merged = cv2.merge(ch)
assert merged.shape == view.shape

diff = cv2.absdiff(merged, view)
assert cv2.norm(diff, cv2.NORM_INF) == 0.0
print("ok")
"""
    p = _run(code)
    assert p.returncode == 0, p.stderr
    assert "ok" in p.stdout


def test_no_crash_fuzz_bitwise_threshold_compare():
    code = r"""
import numpy as np
import cv2

rng = np.random.default_rng(0)

for _ in range(200):
    h = int(rng.integers(1, 16))
    w = int(rng.integers(1, 16))
    a = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    b = rng.integers(0, 256, size=(h, w), dtype=np.uint8)

    # Make strided/non-contiguous views sometimes.
    if rng.random() < 0.5:
        a = a[::2, ::2]
        b = b[::2, ::2]

    # In-place bitwise_not should never crash.
    c = a.copy()
    cv2.bitwise_not(c, c)

    # Masked bitwise ops should never crash.
    mask = rng.integers(0, 2, size=a.shape, dtype=np.uint8) * 255
    dst = np.zeros_like(a)
    cv2.bitwise_and(a, b, dst, mask)

    # Threshold should never crash.
    _r, t = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
    assert t.shape == a.shape

    # Compare should never crash.
    m = cv2.compare(a, b, cv2.CMP_LE)
    assert m.shape == a.shape

print('ok')
"""
    p = _run(code)
    assert p.returncode == 0, p.stderr
    assert "ok" in p.stdout


def test_no_crash_fuzz_inrange_copyto():
    code = r"""
import numpy as np
import cv2

rng = np.random.default_rng(1)

for _ in range(200):
    h = int(rng.integers(1, 16))
    w = int(rng.integers(1, 16))
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    # Sometimes use strided view.
    if rng.random() < 0.5:
        img = img[::2, ::2, :]

    lo = [int(rng.integers(0, 128))] * 3
    hi = [int(rng.integers(128, 256))] * 3
    mask = cv2.inRange(img, lo, hi)
    assert mask.shape == img.shape[:2]

    dst = np.full_like(img, 255)
    cv2.copyTo(img, mask, dst)

print('ok')
"""
    p = _run(code)
    assert p.returncode == 0, p.stderr
    assert "ok" in p.stdout


def test_no_crash_fuzz_filtering():
    code = r"""
import numpy as np
import cv2

rng = np.random.default_rng(2)

for _ in range(100):
    h = int(rng.integers(1, 20))
    w = int(rng.integers(1, 20))
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    if rng.random() < 0.5:
        img = img[::2, ::2, :]

    kw = int(rng.integers(1, 8)) | 1
    kh = int(rng.integers(1, 8)) | 1
    out1 = cv2.blur(img, (kw, kh))
    assert out1.shape == img.shape

    sigx = float(rng.random() * 2.0 + 0.1)
    sigy = float(rng.random() * 2.0 + 0.1)
    out2 = cv2.GaussianBlur(img, (kw, kh), sigx, sigmaY=sigy)
    assert out2.shape == img.shape

print('ok')
"""
    p = _run(code)
    assert p.returncode == 0, p.stderr
    assert "ok" in p.stdout


def test_no_crash_fuzz_derivatives():
    code = r"""
import numpy as np
import cv2

rng = np.random.default_rng(3)

for _ in range(150):
    h = int(rng.integers(1, 24))
    w = int(rng.integers(1, 24))
    img = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    if rng.random() < 0.5:
        img = img[::2, ::2]

    out1 = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    assert out1.shape == img.shape
    out2 = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    assert out2.shape == img.shape

print('ok')
"""
    p = _run(code)
    assert p.returncode == 0, p.stderr
    assert "ok" in p.stdout


def test_no_crash_fuzz_canny():
    code = r"""
import numpy as np
import cv2

rng = np.random.default_rng(4)

for _ in range(100):
    h = int(rng.integers(1, 40))
    w = int(rng.integers(1, 40))
    if rng.random() < 0.5:
        img = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        if rng.random() < 0.5:
            img = img[::2, ::2]
    else:
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        if rng.random() < 0.5:
            img = img[::2, ::2, :]

    t1 = float(rng.random() * 50.0)
    t2 = float(rng.random() * 150.0)
    edges = cv2.Canny(img, t1, t2)
    assert edges.shape == img.shape[:2]

print('ok')
"""
    p = _run(code)
    assert p.returncode == 0, p.stderr
    assert "ok" in p.stdout
