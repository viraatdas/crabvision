#!/usr/bin/env bash
set -euo pipefail

# Sync a *sparse* checkout of OpenCV + opencv_extra into ./vendor for local test/dev use.
#
# Why sparse? Full OpenCV clones are huge. For now we only need:
# - OpenCV: Python test suite sources (modules/python/test)
# - opencv_extra: test data referenced by those tests
#
# Notes:
# - Upstream tests will not pass until we implement enough `cv2` API.
# - This repo intentionally does NOT commit vendor/ contents; see .gitignore.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

OPENCV_DIR="$ROOT_DIR/vendor/opencv"
OPENCV_EXTRA_DIR="$ROOT_DIR/vendor/opencv_extra"

OPENCV_URL="https://github.com/opencv/opencv.git"
OPENCV_EXTRA_URL="https://github.com/opencv/opencv_extra.git"

# Default to the OpenCV maintenance branch.
OPENCV_REF="${OPENCV_REF:-4.x}"
OPENCV_EXTRA_REF="${OPENCV_EXTRA_REF:-4.x}"

sync_sparse_repo() {
  local url="$1"
  local dir="$2"
  local ref="$3"
  shift 3

  if [ -d "$dir/.git" ]; then
    echo "Updating $dir ($ref)..."
    git -C "$dir" fetch --depth=1 origin "$ref"
    git -C "$dir" checkout -f FETCH_HEAD
  else
    echo "Cloning $url -> $dir ($ref)..."
    rm -rf "$dir"
    git clone --depth=1 --filter=blob:none --sparse --branch "$ref" "$url" "$dir"
  fi

  # Ensure sparse paths are set each run (lets us evolve the list over time).
  git -C "$dir" sparse-checkout set --cone "$@"
  git -C "$dir" checkout -f
}

mkdir -p "$ROOT_DIR/vendor"

sync_sparse_repo "$OPENCV_URL" "$OPENCV_DIR" "$OPENCV_REF" \
  modules/python/test

# opencv_extra contains most of the data that OpenCV's Python tests reference via
# OPENCV_TEST_DATA_PATH (e.g. cv/shared/lena.png).
sync_sparse_repo "$OPENCV_EXTRA_URL" "$OPENCV_EXTRA_DIR" "$OPENCV_EXTRA_REF" \
  testdata/cv/shared \
  testdata/highgui/readwrite

echo "Synced OpenCV into $OPENCV_DIR"
echo "Synced opencv_extra into $OPENCV_EXTRA_DIR"
echo "To use test data: export OPENCV_TEST_DATA_PATH=$OPENCV_EXTRA_DIR/testdata"
