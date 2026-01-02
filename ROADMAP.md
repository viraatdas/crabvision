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
- [ ] Decide on alias/meta-package name (optional; currently not shipped)
- [x] Versioning policy + changelog
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
- [x] `norm` subset (INF/L1/L2 for common numeric dtypes)
- [x] bitwise ops subset (with optional `dst`)
- [x] `compare` subset (CMP_* producing 0/255 masks)
- [x] `countNonZero`
- [x] Mask support on bitwise ops (OpenCV-style `mask=`)

### Color + geometry

- [x] `cvtColor` subset
- [x] `resize` subset
- [x] `split/merge`
- [x] `flip`, `rotate`, `transpose`
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

### Morphology

- [x] `getStructuringElement`, `erode`, `dilate`, `morphologyEx` (uint8, basic ops)

### Contours / features (longer-term)

- [ ] `findContours` (hard; large surface)
- [ ] keypoints/descriptors (ORB/BRISK)

## 4. Performance

- [ ] Avoid extra allocations (where safe)
- [ ] SIMD/packed loops for hot paths
- [ ] Optional multi-threading (rayon) with careful GIL interaction

## 5. Packaging goals

- [ ] Installable as `crabvision` (provides `import cv2`)
- [ ] Optional alias meta-package (name TBD)

---

# OpenCV Parity Roadmap (Long-Term)

This section describes what’s missing to reach **practical parity** with upstream OpenCV’s Python API (`cv2`).

Important reality check:

- **“Complete parity”** with upstream OpenCV (all modules, all dtypes, all flags, all numerical corner cases, all performance characteristics, CUDA/OpenCL backends, platform-specific quirks) is a **multi‑year** effort.
- What we *can* do is drive toward **module-by-module parity** with objective gates: upstream tests + data + compatibility checklists.

This roadmap is structured so progress is measurable and can be executed incrementally.

## Parity definition

Track parity across four axes:

1) **API surface parity**: function names, signatures, constants, classes.
2) **Behavior parity**: return types (`None` vs exception), dtype/shape rules, broadcasting rules, in-place semantics, `dst` behavior.
3) **Numerical parity**: outputs match within tolerance (or exactly where required), deterministic behavior when upstream is deterministic.
4) **Operational parity**: no interpreter crashes, reasonable performance, stable packaging.

Recommended target:

- **Behavior parity** + **no-crash guarantees** first.
- Numerical parity within documented tolerances.
- Performance parity later.

## Parity gates

To claim parity for an area, require:

- ✅ **Upstream-derived tests**: port or directly run a curated subset of OpenCV’s Python tests.
- ✅ **Test data**: `opencv_extra` (and eventually `opencv_contrib` test data if needed).
- ✅ **No-crash subprocess tests**: randomized inputs in a subprocess (detect segfault/abort).
- ✅ **CI coverage**: macOS/Linux/Windows, CPython 3.8–3.12 wheels.

## What remains for “complete parity”

### Phase A — Core/Imgproc “everyday parity” (highest leverage)

**A1. Dtypes + scalar rules (huge parity booster)**

- [ ] Expand dtype support beyond `uint8`:
  - [ ] `int16`, `uint16`, `int32`, `float32`, `float64`
  - [ ] define saturation vs wrap vs float behavior per op
- [ ] Scalar handling parity: allow Python ints/floats in most arithmetic ops
- [ ] Consistent `dst` support across more functions (`dst` dtype validation, shape checks)
- [ ] Non-contiguous array handling: ensure all ops accept strided arrays

**A2. Border types and ROI semantics**

- [ ] Implement border modes consistently:
  - [ ] `BORDER_CONSTANT`, `BORDER_REPLICATE`, `BORDER_REFLECT`, `BORDER_REFLECT_101`, `BORDER_WRAP`
- [ ] ROI slicing behavior parity (views, writes into `dst` views)

**A3. Additional fundamental transforms**

- [ ] `copyMakeBorder`
- [ ] `warpAffine` (at least nearest/linear)
- [ ] `warpPerspective`
- [ ] `remap`
- [ ] `getAffineTransform`, `getPerspectiveTransform`

**A4. Filtering / convolution family**

- [ ] `boxFilter` (explicit), `filter2D`
- [ ] `medianBlur`, `bilateralFilter`
- [ ] `Laplacian`
- [ ] `Sobel/Scharr` complete: `ksize` variants, `dx/dy` ranges, `borderType` variants

**A5. Morphology complete**

- [ ] `morphologyEx` full parameter parity: `anchor`, `borderType`, `borderValue`
- [ ] `getStructuringElement` exact ellipse behavior parity (shape details)
- [ ] `distanceTransform`, `watershed` (later)

**A6. Color conversions**

- [ ] Expand `cvtColor` conversions and constant table (HSV, HLS, YCrCb, Lab, etc.)
- [ ] `split/merge` full dtype support

**A7. Histogram / stats**

- [ ] `calcHist`
- [ ] `equalizeHist`
- [ ] `mean`, `meanStdDev`
- [ ] `minMaxLoc`

### Phase B — Shapes, drawing, and higher-level primitives

- [ ] Drawing API: `line`, `circle`, `rectangle`, `putText`
- [ ] `findContours`, `drawContours` (big surface)
- [ ] Shape ops: `boundingRect`, `contourArea`, `arcLength`, `approxPolyDP`
- [ ] `HoughLines`, `HoughLinesP`, `HoughCircles`

### Phase C — Features2D and matching

- [ ] ORB/BRISK/FAST detectors
- [ ] BFMatcher / FLANN wrapper parity
- [ ] KeyPoint/Descriptor data model parity

### Phase D — Calib3d

- [ ] `findChessboardCorners`, `calibrateCamera`, `solvePnP`
- [ ] `stereoCalibrate`, `stereoRectify`

### Phase E — VideoIO / HighGUI

- [ ] `VideoCapture` + `VideoWriter` (platform dependent)
- [ ] `imshow`, `waitKey` (likely optional / headless stubs)

### Phase F — DNN and contrib modules

- [ ] DNN (`cv2.dnn`) inference API
- [ ] opencv_contrib modules (xfeatures2d, aruco, etc.)

## Parity release criteria

If we ever declare parity for a scope, it must be scoped and measurable, e.g.:

- “Parity with OpenCV `core+imgproc` (subset)”

And must satisfy:

- [ ] ≥ N upstream-derived tests passing (tracked list)
- [ ] CI green on macOS/Linux/Windows
- [ ] Wheels for cp38–cp312
