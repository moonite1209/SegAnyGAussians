"""
Tests for CameraLoader module.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene.camera_spec import CameraSpec
from scene.camera_data import CameraData
from utils.camera_loader import CameraLoader, DataLoadError


def test_camera_loader_basic():
    """Test basic CameraLoader functionality."""
    print("Testing CameraLoader basic functionality...")

    # Create a mock CameraSpec
    spec = CameraSpec(
        uid=0,
        image_name="test_image.jpg",
        width=1920,
        height=1080,
        fx=1000.0,
        fy=1000.0,
        cx=960.0,
        cy=540.0,
        R=np.eye(3),
        T=np.zeros(3),
        image_path=Path("data/garden/images/00000_00.jpg"),
        mask_path=None,
        labels_path=None,
        depth_path=None,
        confidence_path=None
    )

    # Test that loader can be created
    loader = CameraLoader(resolution_scale=1.0, resolution=1)
    assert loader.resolution_scale == 1.0
    assert loader.resolution == 1

    # Test resolution calculation
    w, h = loader._calculate_target_resolution(1920, 1080)
    assert w == 1920 and h == 1080  # Full resolution

    # Test with resolution=2
    loader2 = CameraLoader(resolution_scale=1.0, resolution=2)
    w2, h2 = loader2._calculate_target_resolution(1920, 1080)
    assert w2 == 960 and h2 == 540  # Half resolution

    print("✓ CameraLoader basic test passed")


def test_camera_loader_resolution_calculation():
    """Test resolution calculation with different settings."""
    print("Testing CameraLoader resolution calculation...")

    loader = CameraLoader(resolution_scale=1.0, resolution=1)

    # Test standard resolutions
    test_cases = [
        (1920, 1080, 1, 1920, 1080),
        (1920, 1080, 2, 960, 540),
        (1920, 1080, 4, 480, 270),
        (1920, 1080, 8, 240, 135),
        (1920, 1080, -1, 1600, 900),  # Auto-scale for large images
    ]

    for orig_w, orig_h, res, expected_w, expected_h in test_cases:
        loader.resolution = res
        w, h = loader._calculate_target_resolution(orig_w, orig_h)
        assert w == expected_w and h == expected_h, \
            f"Failed for resolution={res}: expected ({expected_w}, {expected_h}), got ({w}, {h})"

    # Test resolution_scale
    loader2 = CameraLoader(resolution_scale=0.5, resolution=1)
    w, h = loader2._calculate_target_resolution(1920, 1080)
    assert w == 3840 and h == 2160  # Scale 0.5 means 2x resolution

    print("✓ Resolution calculation test passed")


def test_data_load_error():
    """Test DataLoadError exception."""
    print("Testing DataLoadError...")

    error = DataLoadError(Path("/fake/path.jpg"), "File not found")
    assert str(error) == "Failed to load /fake/path.jpg: File not found"
    assert error.path == Path("/fake/path.jpg")
    assert error.reason == "File not found"

    print("✓ DataLoadError test passed")


def test_load_nonexistent_image():
    """Test that loading a non-existent image raises DataLoadError."""
    print("Testing loading of non-existent image...")

    spec = CameraSpec(
        uid=0,
        image_name="nonexistent.jpg",
        width=1920,
        height=1080,
        fx=1000.0,
        fy=1000.0,
        cx=960.0,
        cy=540.0,
        R=np.eye(3),
        T=np.zeros(3),
        image_path=Path("/nonexistent/path/image.jpg"),
        mask_path=None,
        labels_path=None,
        depth_path=None,
        confidence_path=None
    )

    loader = CameraLoader(resolution_scale=1.0, resolution=1)

    try:
        loader.load(spec)
        assert False, "Should have raised DataLoadError"
    except DataLoadError as e:
        assert "not found" in str(e).lower()
        print(f"  Correctly raised error: {e}")

    print("✓ Non-existent image test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running CameraLoader Tests")
    print("="*60 + "\n")

    try:
        test_camera_loader_basic()
        test_camera_loader_resolution_calculation()
        test_data_load_error()
        test_load_nonexistent_image()

        print("\n" + "="*60)
        print("All CameraLoader tests passed! ✓")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
