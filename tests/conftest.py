import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--opencv-upstream",
        action="store_true",
        help="Run OpenCV-upstream-derived tests (requires vendored test data)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--opencv-upstream"):
        # By default skip any upstream-derived tests which may require additional data.
        skip_up = pytest.mark.skip(reason="use --opencv-upstream to run upstream-derived tests")
        for item in items:
            if item.get_closest_marker("opencv_upstream") is not None:
                item.add_marker(skip_up)
                continue

            # If the user points pytest directly at a vendored checkout of OpenCV,
            # avoid collecting it by default.
            if str(item.fspath).startswith(os.path.abspath("vendor/opencv")):
                item.add_marker(skip_up)
