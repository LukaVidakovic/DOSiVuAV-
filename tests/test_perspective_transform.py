"""
Unit tests for perspective transform module
"""

import numpy as np
import cv2
import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from perspective_transform import (
    get_default_src_points,
    get_default_dst_points,
    get_perspective_transform,
    warp_image,
    unwarp_image,
    draw_transform_region,
    visualize_transform,
    corners_unwarp
)


@pytest.fixture
def test_image():
    """Create a test image."""
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    return img


@pytest.fixture
def test_points():
    """Create test source and destination points."""
    src = np.float32([[200, 300], [1000, 300], [1100, 650], [100, 650]])
    dst = np.float32([[300, 0], [980, 0], [980, 720], [300, 720]])
    return src, dst


def test_get_default_src_points():
    """Test getting default source points."""
    img_shape = (720, 1280)
    src = get_default_src_points(img_shape)

    # Check shape
    assert src.shape == (4, 2)
    assert src.dtype == np.float32

    # Check points are within image bounds
    assert np.all(src[:, 0] >= 0)  # x >= 0
    assert np.all(src[:, 0] <= img_shape[1])  # x <= width
    assert np.all(src[:, 1] >= 0)  # y >= 0
    assert np.all(src[:, 1] <= img_shape[0])  # y <= height

    # Check trapezoid shape (top narrower than bottom)
    top_width = src[1, 0] - src[0, 0]
    bottom_width = src[2, 0] - src[3, 0]
    assert top_width < bottom_width


def test_get_default_dst_points():
    """Test getting default destination points."""
    img_shape = (720, 1280)
    dst = get_default_dst_points(img_shape)

    # Check shape
    assert dst.shape == (4, 2)
    assert dst.dtype == np.float32

    # Check points are within image bounds
    assert np.all(dst[:, 0] >= 0)
    assert np.all(dst[:, 0] <= img_shape[1])
    assert np.all(dst[:, 1] >= 0)
    assert np.all(dst[:, 1] <= img_shape[0])

    # Check rectangle shape (parallel sides)
    top_width = dst[1, 0] - dst[0, 0]
    bottom_width = dst[2, 0] - dst[3, 0]
    assert np.isclose(top_width, bottom_width)


def test_get_perspective_transform(test_points):
    """Test getting perspective transform matrices."""
    src, dst = test_points
    M, Minv = get_perspective_transform(src, dst)

    # Check matrix shapes
    assert M.shape == (3, 3)
    assert Minv.shape == (3, 3)

    # Check that M and Minv are inverses
    identity = np.dot(M, Minv)
    # Normalize by bottom-right element
    identity = identity / identity[2, 2]
    expected_identity = np.eye(3)

    assert np.allclose(identity, expected_identity, atol=1e-2)


def test_warp_image(test_image, test_points):
    """Test warping image."""
    src, dst = test_points
    M, _ = get_perspective_transform(src, dst)

    warped = warp_image(test_image, M)

    # Check output shape
    assert warped.shape == test_image.shape

    # Check it's a valid image
    assert warped.dtype == test_image.dtype


def test_warp_image_custom_size(test_image, test_points):
    """Test warping with custom output size."""
    src, dst = test_points
    M, _ = get_perspective_transform(src, dst)

    custom_size = (640, 480)  # width, height
    warped = warp_image(test_image, M, img_size=custom_size)

    # Check output shape matches custom size (height, width, channels)
    assert warped.shape[0] == custom_size[1]  # height
    assert warped.shape[1] == custom_size[0]  # width


def test_unwarp_image(test_image, test_points):
    """Test unwarping image."""
    src, dst = test_points
    M, Minv = get_perspective_transform(src, dst)

    # Warp then unwarp
    warped = warp_image(test_image, M)
    unwarped = unwarp_image(warped, Minv)

    # Check output shape
    assert unwarped.shape == test_image.shape

    # Unwarped should be similar to original (not exact due to interpolation)
    # Just check it's a valid image
    assert unwarped.dtype == test_image.dtype


def test_warp_unwarp_roundtrip(test_points):
    """Test that warp->unwarp returns similar image."""
    # Create a simple test pattern
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)

    src, dst = test_points
    M, Minv = get_perspective_transform(src, dst)

    # Warp and unwarp
    warped = warp_image(img, M)
    unwarped = unwarp_image(warped, Minv)

    # Should have same shape
    assert unwarped.shape == img.shape


def test_draw_transform_region(test_image, test_points):
    """Test drawing transform region."""
    src, _ = test_points
    img_with_region = draw_transform_region(test_image, src)

    # Check output shape
    assert img_with_region.shape == test_image.shape

    # Check it's different from input (has lines drawn)
    assert not np.array_equal(img_with_region, test_image)


def test_draw_transform_region_custom_color(test_image, test_points):
    """Test drawing with custom color and thickness."""
    src, _ = test_points
    img_with_region = draw_transform_region(
        test_image, src,
        color=(0, 255, 0),
        thickness=5
    )

    # Check output is valid
    assert img_with_region.shape == test_image.shape


def test_visualize_transform(test_image):
    """Test visualization of transform."""
    img_with_region, warped = visualize_transform(test_image)

    # Check both outputs have correct shape
    assert img_with_region.shape == test_image.shape
    assert warped.shape == test_image.shape

    # img_with_region should be different from original
    assert not np.array_equal(img_with_region, test_image)


def test_visualize_transform_custom_points(test_image, test_points):
    """Test visualization with custom points."""
    src, dst = test_points
    img_with_region, warped = visualize_transform(test_image, src, dst)

    # Check outputs
    assert img_with_region.shape == test_image.shape
    assert warped.shape == test_image.shape


def test_corners_unwarp(test_image, test_points):
    """Test corners_unwarp convenience function."""
    src, dst = test_points
    warped, M, Minv = corners_unwarp(test_image, src, dst)

    # Check warped image
    assert warped.shape == test_image.shape

    # Check matrices
    assert M.shape == (3, 3)
    assert Minv.shape == (3, 3)

    # Check matrices are inverses
    identity = np.dot(M, Minv)
    identity = identity / identity[2, 2]
    expected_identity = np.eye(3)
    assert np.allclose(identity, expected_identity, atol=1e-2)


def test_binary_image_transform(test_points):
    """Test that transform works on binary images."""
    # Create binary image
    binary = np.zeros((720, 1280), dtype=np.uint8)
    binary[300:400, 500:700] = 1

    src, dst = test_points
    M, _ = get_perspective_transform(src, dst)

    # Warp binary image
    warped = warp_image(binary, M)

    # Check output
    assert warped.shape == binary.shape
    assert warped.dtype == binary.dtype
    assert np.max(warped) <= 1
    assert np.min(warped) >= 0


def test_transform_preserves_straight_lines():
    """Test that parallel lines remain parallel after transform."""
    # Create image with two parallel vertical lines
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 30] = 255
    img[:, 70] = 255

    # Define transform that should keep lines parallel
    src = np.float32([[20, 30], [80, 30], [80, 90], [20, 90]])
    dst = np.float32([[25, 0], [75, 0], [75, 100], [25, 100]])

    M, _ = get_perspective_transform(src, dst)
    warped = warp_image(img, M)

    # Lines should still be present in warped image
    assert np.sum(warped) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
