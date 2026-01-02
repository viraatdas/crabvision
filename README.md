# Crabvision (opencv with a rust backend)

![OpenCV Parity: partial](https://img.shields.io/badge/OpenCV%20parity-partial-yellow)

A Rust-native, safe subset of OpenCV's Python API exposed as a `cv2` module.

This is an MVP that focuses on correctness and memory safety by implementing core image ops in Rust via PyO3:

- imread / imwrite (PNG, JPEG)
- imdecode / imencode (PNG, JPEG)
- cvtColor: BGR<->RGB, BGR->GRAY, GRAY->BGR
- resize: INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4
- absdiff, add, subtract (uint8, saturating)
- split / merge (3-channel uint8)
- norm: NORM_INF / NORM_L1 / NORM_L2 (uint8)
- bitwise_not / bitwise_and / bitwise_or / bitwise_xor (optional `dst` and optional `mask`)
- threshold: THRESH_BINARY / THRESH_BINARY_INV / THRESH_TRUNC / THRESH_TOZERO / THRESH_TOZERO_INV
- compare: CMP_EQ / CMP_NE / CMP_LT / CMP_LE / CMP_GT / CMP_GE
- countNonZero
- inRange
- copyTo (masked copy)
- blur (box filter)
- GaussianBlur
- Sobel / Scharr
- Canny
- flip / transpose / rotate
- getStructuringElement / erode / dilate / morphologyEx

Distribution name is `crabvision`, but you `import cv2` like with OpenCV.

## OpenCV parity

This project is **not** in full parity with upstream OpenCV.

- Goal: grow a safe, well-tested subset of the `cv2` API.
- Status: **partial parity** (core I/O + common image ops + filtering + edges; many modules like features2d, calib3d, video, etc. are not implemented).
- Tracking: see `ROADMAP.md` and the `tests/test_upstream_ported_*.py` suite.

If you want “complete parity”, the practical path is module-by-module (Core → Imgproc → Highgui → …), continuously validated by upstream-derived tests.

## Install (local dev) with uv

- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Create a virtualenv (PyO3 0.21 supports CPython up to 3.12):

```bash
uv venv -p python3.12
```

- Build and install in editable mode:

```bash
uv pip install -e .
```

uv will use the maturin build backend to compile the Rust extension and install a wheel named `crabvision`, which exposes a module `cv2`.

## Install from PyPI

Once published, install with:

```bash
pip install crabvision
```

Then use it with:

```python
import cv2
print(cv2.__version__)
```

Alternatively, build wheels:

```bash
uv build
```

## Usage

```python
import cv2
import numpy as np

img = np.zeros((64, 64, 3), dtype=np.uint8)
img[..., 2] = 255  # red in BGR
res = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
cv2.imwrite("out.png", gray)
```

## Upstream tests (starting point)

This repo is intended to grow by porting and reusing OpenCV's Python tests.

- Sync upstream sources and test data (sparse checkout into `vendor/`):

```bash
./scripts/sync_upstream_tests.sh
```

- Run the upstream-derived (ported) tests:

```bash
export OPENCV_TEST_DATA_PATH=vendor/opencv_extra/testdata
uv run pytest --opencv-upstream
```

## Safety regression tests (no hard crashes)

OpenCV’s native backend can occasionally take down the Python process in the presence of
bad inputs (e.g., shape/dtype mismatches, non-contiguous arrays) because it crosses a C/C++ ABI.

This repo includes a small set of “no hard crash” tests that run `cv2` calls in a subprocess.
If anything segfaults (or otherwise aborts the interpreter), the subprocess exits non-zero and the
test fails.

- File: `tests/test_safety_no_crash.py`
- Run: `uv run pytest -k safety_no_crash`

## Releases (PyPI)

This repo includes GitHub Actions workflows to build and publish `crabvision` (the Rust extension that provides `import cv2`).

Release workflow: `.github/workflows/release.yml`.

Recommended setup is PyPI “Trusted Publishing” (OIDC), so you don’t need to store a PyPI API token.

- In PyPI, add a Trusted Publisher for your project(s) pointing at this GitHub repo.
- Create a git tag like `v0.0.3` and push it; the workflow will build wheels/sdists and publish.

## Changelog

See `CHANGELOG.md`.

## Scope and roadmap

This is not a full rewrite of OpenCV. The long-term aim is to grow a safe, well-tested subset of the API with predictable performance. Next steps could include:

- More color conversions and image formats
- Filtering (GaussianBlur, medianBlur), morphology, edges (Canny)
- Video IO via `ffmpeg` bindings
- Basic feature ops

Contributions welcome.

## License

MIT
