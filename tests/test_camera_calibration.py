"""
Unit tests for camera calibration module
"""

import numpy as np
import cv2
import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from camera_calibration import (
    calibrate_camera,
    save_calibration,
    load_calibration,
    undistort_image
)


def test_calibrate_camera():
    """Test camera calibration with chessboard images."""
    # This test requires actual calibration images
    if not os.path.exists("camera_cal"):
        pytest.skip("Camera calibration images not found")

    mtx, dist, rvecs, tvecs = calibrate_camera(
        images_path="camera_cal/calibration*.jpg",
        chessboard_size=(9, 6),
        save_path=None  # Don't save during test
    )

    # Check that calibration returns valid matrices
    assert mtx is not None
    assert dist is not None
    assert mtx.shape == (3, 3)
    # Distortion coefficients: (5,) or (5, 1) or (1, 5)
    assert len(dist.flatten()) >= 4  # At least 4 distortion coefficients

    # Camera matrix should have non-zero focal lengths
    assert mtx[0, 0] > 0  # fx
    assert mtx[1, 1] > 0  # fy


def test_save_and_load_calibration():
    """Test saving and loading calibration data."""
    # Create dummy calibration data
    mtx = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
    dist = np.array([0.1, -0.2, 0.001, 0.002, 0.05], dtype=np.float32)
    rvecs = np.array([[[0.1]], [[0.2]], [[0.3]]], dtype=np.float32)
    tvecs = np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)

    # Save
    test_path = "test_calibration.npz"
    save_calibration(mtx, dist, rvecs, tvecs, test_path)

    # Check file exists
    assert os.path.exists(test_path)

    # Load
    mtx_loaded, dist_loaded, rvecs_loaded, tvecs_loaded = load_calibration(test_path)

    # Verify loaded data matches
    np.testing.assert_array_almost_equal(mtx, mtx_loaded)
    np.testing.assert_array_almost_equal(dist, dist_loaded)

    # Cleanup
    os.remove(test_path)


def test_undistort_image():
    """Test image undistortion."""
    # Create a simple test image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Create dummy camera parameters
    mtx = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Undistort (with zero distortion, should return similar image)
    undistorted = undistort_image(img, mtx, dist, optimize=False)

    # Check output has same shape
    assert undistorted.shape == img.shape

    # With zero distortion, images should be very similar
    assert np.mean(np.abs(img - undistorted)) < 1.0


def test_load_calibration_file_not_found():
    """Test that loading non-existent calibration file raises error."""
    with pytest.raises(FileNotFoundError):
        load_calibration("nonexistent_calibration.npz")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
