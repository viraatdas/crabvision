# OpenCV Parity Roadmap (Long-Term)

This document describes what’s missing to reach **practical parity** with upstream OpenCV’s Python API (`cv2`).

Important reality check:

- **“Complete parity”** with upstream OpenCV (all modules, all dtypes, all flags, all numerical corner cases, all performance characteristics, CUDA/OpenCL backends, platform-specific quirks) is a **multi‑year** effort.
- What we *can* do is drive toward **module-by-module parity** with objective gates: upstream tests + data + compatibility checklists.

This roadmap is structured so progress is measurable and can be executed incrementally.

## 1. Define what “parity” means

We should explicitly track parity across four axes:

1) **API surface parity**: function names, signatures, constants, classes.
2) **Behavior parity**: return types (`None` vs exception), dtype/shape rules, broadcasting rules, in-place semantics, `dst` behavior.
3) **Numerical parity**: outputs match within tolerance (or exactly where required), deterministic behavior when upstream is deterministic.
4) **Operational parity**: no interpreter crashes, reasonable performance, stable packaging.

For this project’s long-term goals, the recommended target is:

- **Behavior parity** + **no-crash guarantees** as top priority.
- Numerical parity within documented tolerances.
- Performance parity is a later-stage optimization.

## 2. Parity gates (how we prove parity)

To claim parity for an area, require:

- ✅ **Upstream-derived tests**: port or directly run a curated subset of OpenCV’s Python tests.
- ✅ **Test data**: `opencv_extra` (and eventually `opencv_contrib` test data if needed).
- ✅ **No-crash subprocess tests**: randomized inputs in a subprocess (detect segfault/abort).
- ✅ **CI coverage**: macOS/Linux/Windows, CPython 3.8–3.12 wheels.

## 3. Current status (high level)

Current implementation is intentionally a **subset**. It already includes:

- I/O: `imread/imwrite`, `imdecode/imencode`
- Core ops: `resize`, `cvtColor` (subset), arithmetic (`add/subtract/absdiff`), bitwise ops (with `dst` + `mask`), `compare`, `norm` subset
- Thresholding: `threshold`, `inRange`, `copyTo`
- Filtering: `blur`, `GaussianBlur`
- Derivatives: `Sobel`, `Scharr`
- Edges: `Canny`
- Geometry: `flip`, `transpose`, `rotate`
- Morphology: `getStructuringElement`, `erode`, `dilate`, `morphologyEx`

Everything below is what remains for “complete parity”.

## 4. Phase plan (recommended order)

### Phase A — Core/Imgproc “everyday parity” (highest leverage)

Goal: cover the majority of common image-processing scripts.

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

### Phase B — Shapes, drawing, and high-level primitives

Goal: enable a large class of “computer vision tutorial” scripts.

- [ ] Drawing API: `line`, `circle`, `rectangle`, `putText`
- [ ] `findContours`, `drawContours` (big surface)
- [ ] Shape ops: `boundingRect`, `contourArea`, `arcLength`, `approxPolyDP`
- [ ] `HoughLines`, `HoughLinesP`, `HoughCircles`

### Phase C — Features2D and matching

Goal: feature detection + matching scripts.

- [ ] ORB/BRISK/FAST detectors
- [ ] BFMatcher / FLANN wrapper parity
- [ ] KeyPoint/Descriptor data model parity

### Phase D — Calib3d

Goal: camera geometry and pose.

- [ ] `findChessboardCorners`, `calibrateCamera`, `solvePnP`
- [ ] `stereoCalibrate`, `stereoRectify`

### Phase E — VideoIO / HighGUI

Goal: basic video capture and display.

- [ ] `VideoCapture` + `VideoWriter` (platform dependent)
- [ ] `imshow`, `waitKey` (likely optional / headless stubs)

### Phase F — DNN and contrib modules (very large)

Goal: parity with deep learning and contributed modules.

- [ ] DNN (`cv2.dnn`) inference API
- [ ] opencv_contrib modules (xfeatures2d, aruco, etc.)

## 5. Test plan to reach parity

To keep parity claims honest, grow test coverage in layers:

1) **Unit tests** for each function (shape/dtype/edge cases).
2) **Upstream-derived tests** for behavior parity.
3) **Golden-data tests** (hashes) only where stable and justified.
4) **Subprocess fuzz** for every new primitive.

## 6. Deliverables checklist for “Parity Release” (definition)

If we ever declare “parity” for a scope, it should be scoped and named, e.g.:

- “Parity with OpenCV `core+imgproc` (subset)”

And that release must satisfy:

- [ ] ≥ N upstream-derived tests passing (tracked list)
- [ ] CI green on macOS/Linux/Windows
- [ ] Wheels for cp38–cp312
- [ ] No-crash fuzz suites pass

