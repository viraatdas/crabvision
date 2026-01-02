# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows SemVer.

## [0.0.4] - 2026-01-02

### Fixed
- Release automation: build/publish now uploads the actual wheel artifacts and avoids 32-bit Linux (i686) builds that fail in manylinux.

## [0.0.3] - 2026-01-02

### Added
- Core image ops and safety tests: bitwise ops, thresholding, inRange/copyTo, filtering (blur/GaussianBlur), derivatives (Sobel/Scharr), edges (Canny).
- Upstream-derived tests (opt-in via `--opencv-upstream`) and subprocess “no hard crash” fuzz tests.
- GitHub Actions CI and wheel/release workflows.

### Changed
- `cv2.Canny` accepts color images and converts to grayscale internally.
