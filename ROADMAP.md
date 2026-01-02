# CrabVision Roadmap

This project grows by implementing small, safe slices of the `cv2` API in Rust and validating them with:

- Ported / upstream-derived OpenCV Python tests (opt-in)
- Subprocess “no hard crash” tests (detect segfault-like regressions)

Below is a living checklist of what still needs to be done.

## 0. Project hygiene

- [x] CI: build wheels on macOS/Linux/Windows (cibuildwheel + maturin)
- [x] CI: run `pytest` on macOS/Linux/Windows
- [x] CI: run upstream-derived tests (`--opencv-upstream`) on Linux
- [x] Publish to PyPI under `crabvision` (release workflow + trusted publishing)
- [ ] Decide on alias/meta-package name (optional)
- [ ] Versioning policy + changelog
- [ ] Type hints / stubs for `cv2` surface we implement

## 1. Safety strategy (never crash the interpreter)

- [ ] Expand subprocess fuzz tests (random shapes/strides/edge values)
- [ ] Add explicit “bad input” tests for every new API surface
- [ ] Validate no UB: avoid aliasing readonly+readwrite borrows; keep copies where needed
- [ ] Add `RUST_BACKTRACE=1` friendly error paths

## 2. Core array/model compatibility

- [ ] Define supported dtypes per function (start with `uint8`, add `float32`, etc.)
- [ ] Define image memory model: HWC uint8, channel semantics (BGR vs RGB)
- [ ] Support `dst` outputs consistently (shape checks, in-place rules)
- [ ] Support contiguous + non-contiguous views everywhere

## 3. API surface (incremental)

### Image I/O

- [x] `imread/imwrite` (PNG/JPEG)
- [x] `imdecode/imencode` (PNG/JPEG)
- [ ] Alpha channel support (RGBA, IMREAD_UNCHANGED)
- [ ] EXIF orientation behavior (match upstream where possible)

### Core math / logic

- [x] `absdiff`, `add`, `subtract` (uint8, saturating)
- [x] `norm` subset (INF/L1/L2 for uint8)
- [x] bitwise ops subset (with optional `dst`)
- [x] `compare` subset (CMP_* producing 0/255 masks)
- [x] `countNonZero`
- [x] Mask support on bitwise ops (OpenCV-style `mask=`)

### Color + geometry

- [x] `cvtColor` subset
- [x] `resize` subset
- [x] `split/merge`
- [ ] `flip`, `rotate`, `transpose`
- [ ] ROI helpers / border modes

### Thresholding / segmentation

- [x] `threshold` subset (basic modes)
- [x] `inRange`
- [x] `copyTo` (masked copy)

### Filtering

- [x] `GaussianBlur` (uint8, replicate border)
- [x] `boxFilter` / `blur` (as `blur`, uint8, replicate border)
- [x] `Sobel` / `Scharr` (uint8 input, BORDER_DEFAULT)
- [x] `Canny` (uint8 single-channel, aperture 3/5)

### Contours / features (longer-term)

- [ ] `findContours` (hard; large surface)
- [ ] keypoints/descriptors (ORB/BRISK)

## 4. Performance

- [ ] Avoid extra allocations (where safe)
- [ ] SIMD/packed loops for hot paths
- [ ] Optional multi-threading (rayon) with careful GIL interaction

## 5. Packaging goals

- [ ] Installable as `crabvision` (provides `import cv2`)
- [x] Installable alias `opencv-rust` meta package
- [ ] Optional alias `crabvision` vs `opencv-rust` naming conventions in docs
