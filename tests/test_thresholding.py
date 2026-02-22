"""
Unit tests for thresholding module
"""

import numpy as np
import cv2
import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thresholding import (
    abs_sobel_thresh,
    mag_thresh,
    dir_thresh,
    hls_select,
    rgb_select,
    combined_threshold,
    region_of_interest
)


@pytest.fixture
def test_image():
    """Create a simple test image."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return img


@pytest.fixture
def gradient_image():
    """Create an image with vertical gradient for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create vertical lines (should be detected by Sobel X)
    img[:, 30:32] = 255
    img[:, 70:72] = 255
    return img


def test_abs_sobel_thresh_x(gradient_image):
    """Test Sobel X threshold on vertical lines."""
    # Use lower threshold to detect edges
    binary = abs_sobel_thresh(gradient_image, orient='x', sobel_kernel=3, thresh=(10, 255))

    # Check output shape and type
    assert binary.shape == gradient_image.shape[:2]
    assert binary.dtype == np.uint8
    assert np.max(binary) <= 1
    assert np.min(binary) >= 0

    # Should detect some edges (vertical lines should trigger Sobel X)
    assert np.sum(binary) > 0


def test_abs_sobel_thresh_y(gradient_image):
    """Test Sobel Y threshold on vertical lines."""
    binary = abs_sobel_thresh(gradient_image, orient='y', sobel_kernel=3, thresh=(20, 100))

    # Check output shape and type
    assert binary.shape == gradient_image.shape[:2]
    assert binary.dtype == np.uint8


def test_mag_thresh(gradient_image):
    """Test magnitude threshold."""
    binary = mag_thresh(gradient_image, sobel_kernel=3, thresh=(20, 100))

    # Check output shape and type
    assert binary.shape == gradient_image.shape[:2]
    assert binary.dtype == np.uint8
    assert np.max(binary) <= 1
    assert np.min(binary) >= 0


def test_dir_thresh(gradient_image):
    """Test direction threshold."""
    binary = dir_thresh(gradient_image, sobel_kernel=3, thresh=(0, np.pi/2))

    # Check output shape and type
    assert binary.shape == gradient_image.shape[:2]
    assert binary.dtype == np.uint8
    assert np.max(binary) <= 1
    assert np.min(binary) >= 0


def test_hls_select(test_image):
    """Test HLS color space threshold."""
    # Test S-channel (most common for lane detection)
    binary = hls_select(test_image, channel='s', thresh=(100, 255))

    # Check output shape and type
    assert binary.shape == test_image.shape[:2]
    assert binary.dtype == np.uint8
    assert np.max(binary) <= 1
    assert np.min(binary) >= 0


def test_hls_select_all_channels(test_image):
    """Test all HLS channels."""
    for channel in ['h', 'l', 's']:
        binary = hls_select(test_image, channel=channel, thresh=(50, 200))
        assert binary.shape == test_image.shape[:2]


def test_rgb_select(test_image):
    """Test RGB color space threshold."""
    # Test R-channel
    binary = rgb_select(test_image, channel='r', thresh=(100, 255))

    # Check output shape and type
    assert binary.shape == test_image.shape[:2]
    assert binary.dtype == np.uint8
    assert np.max(binary) <= 1
    assert np.min(binary) >= 0


def test_rgb_select_all_channels(test_image):
    """Test all RGB channels."""
    for channel in ['r', 'g', 'b']:
        binary = rgb_select(test_image, channel=channel, thresh=(50, 200))
        assert binary.shape == test_image.shape[:2]


def test_combined_threshold(test_image):
    """Test combined threshold pipeline."""
    binary = combined_threshold(test_image)

    # Check output shape and type
    assert binary.shape == test_image.shape[:2]
    assert binary.dtype == np.uint8
    assert np.max(binary) <= 1
    assert np.min(binary) >= 0


def test_region_of_interest():
    """Test region of interest masking."""
    # Create binary image
    img = np.ones((100, 100), dtype=np.uint8)

    # Define trapezoid region
    vertices = np.array([[(20, 100), (40, 50), (60, 50), (80, 100)]], dtype=np.int32)

    # Apply mask
    masked = region_of_interest(img, vertices)

    # Check output shape
    assert masked.shape == img.shape

    # Check that some pixels are masked out
    assert np.sum(masked) < np.sum(img)

    # Check corners are masked (outside region)
    assert masked[0, 0] == 0
    assert masked[0, 99] == 0


def test_grayscale_input():
    """Test that functions work with grayscale input."""
    gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # These should work with grayscale
    binary = abs_sobel_thresh(gray_img, orient='x')
    assert binary.shape == gray_img.shape

    binary = mag_thresh(gray_img)
    assert binary.shape == gray_img.shape

    binary = dir_thresh(gray_img)
    assert binary.shape == gray_img.shape


def test_threshold_ranges():
    """Test that thresholds produce expected range of outputs."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Very loose threshold should activate many pixels
    binary_loose = abs_sobel_thresh(img, orient='x', thresh=(0, 255))
    pixels_loose = np.sum(binary_loose)

    # Very tight threshold should activate fewer pixels
    binary_tight = abs_sobel_thresh(img, orient='x', thresh=(240, 255))
    pixels_tight = np.sum(binary_tight)

    # Loose threshold should activate more pixels
    assert pixels_loose >= pixels_tight


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
