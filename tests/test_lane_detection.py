"""
Unit tests for lane detection module
"""

import numpy as np
import cv2
import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lane_detection import (
    find_lane_base,
    find_lane_pixels,
    fit_polynomial,
    search_around_poly,
    calculate_curvature,
    calculate_vehicle_position,
    visualize_lane_detection
)


@pytest.fixture
def binary_warped_with_lanes():
    """Create binary warped image with simulated lanes."""
    # Create 720x1280 binary image
    img = np.zeros((720, 1280), dtype=np.uint8)

    # Create left lane (vertical line at x=400)
    img[:, 398:402] = 1

    # Create right lane (vertical line at x=880)
    img[:, 878:882] = 1

    return img


@pytest.fixture
def curved_lane_image():
    """Create binary warped image with curved lanes."""
    img = np.zeros((720, 1280), dtype=np.uint8)

    # Create curved lanes using polynomial
    y = np.linspace(0, 719, 720)

    # Left lane: slight curve
    left_x = 0.0001 * y**2 - 0.05 * y + 400
    for i, yval in enumerate(y):
        x = int(left_x[i])
        if 0 <= x < 1280:
            img[int(yval), max(0, x-2):min(1280, x+2)] = 1

    # Right lane: slight curve
    right_x = 0.0001 * y**2 - 0.05 * y + 880
    for i, yval in enumerate(y):
        x = int(right_x[i])
        if 0 <= x < 1280:
            img[int(yval), max(0, x-2):min(1280, x+2)] = 1

    return img


def test_find_lane_base(binary_warped_with_lanes):
    """Test finding lane base positions using histogram."""
    leftx_base, rightx_base = find_lane_base(binary_warped_with_lanes)

    # Check that bases are detected near expected positions
    assert 390 < leftx_base < 410  # Near x=400
    assert 870 < rightx_base < 890  # Near x=880


def test_find_lane_base_with_visualize(binary_warped_with_lanes):
    """Test finding lane base with histogram output."""
    leftx_base, rightx_base, histogram = find_lane_base(
        binary_warped_with_lanes, visualize=True
    )

    # Check histogram is returned
    assert histogram is not None
    assert len(histogram) == binary_warped_with_lanes.shape[1]

    # Check bases are correct
    assert 390 < leftx_base < 410
    assert 870 < rightx_base < 890


def test_find_lane_pixels(binary_warped_with_lanes):
    """Test sliding window lane pixel detection."""
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped_with_lanes)

    # Check that pixels were found
    assert len(leftx) > 0
    assert len(lefty) > 0
    assert len(rightx) > 0
    assert len(righty) > 0

    # Check arrays have same length
    assert len(leftx) == len(lefty)
    assert len(rightx) == len(righty)

    # Check pixels are in expected x-range
    assert np.mean(leftx) < 500  # Left lane on left side
    assert np.mean(rightx) > 700  # Right lane on right side


def test_find_lane_pixels_parameters():
    """Test sliding window with different parameters."""
    img = np.zeros((720, 1280), dtype=np.uint8)
    img[:, 400:404] = 1  # Thin left lane
    img[:, 880:884] = 1  # Thin right lane

    # Test with different window counts
    leftx, lefty, rightx, righty = find_lane_pixels(
        img, nwindows=12, margin=80, minpix=30
    )

    assert len(leftx) > 0
    assert len(rightx) > 0


def test_fit_polynomial(binary_warped_with_lanes):
    """Test polynomial fitting to lane pixels."""
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped_with_lanes)

    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

    # Check polynomial coefficients shape
    assert len(left_fit) == 3  # [A, B, C]
    assert len(right_fit) == 3

    # For vertical lanes, A and B should be small
    assert abs(left_fit[0]) < 0.001  # Small quadratic term
    assert abs(left_fit[1]) < 0.1    # Small linear term
    assert 390 < left_fit[2] < 410   # C should be near x=400


def test_fit_polynomial_curved(curved_lane_image):
    """Test polynomial fitting on curved lanes."""
    leftx, lefty, rightx, righty = find_lane_pixels(curved_lane_image)

    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

    # Check polynomial coefficients exist
    assert len(left_fit) == 3
    assert len(right_fit) == 3

    # For curved lanes, A should be non-zero
    assert left_fit[0] != 0


def test_search_around_poly(binary_warped_with_lanes):
    """Test optimized search around previous polynomial."""
    # First, get initial polynomial
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped_with_lanes)
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

    # Search around polynomial
    leftx_new, lefty_new, rightx_new, righty_new = search_around_poly(
        binary_warped_with_lanes, left_fit, right_fit, margin=50
    )

    # Should find similar pixels
    assert len(leftx_new) > 0
    assert len(rightx_new) > 0

    # Mean positions should be similar
    assert abs(np.mean(leftx) - np.mean(leftx_new)) < 20
    assert abs(np.mean(rightx) - np.mean(rightx_new)) < 20


def test_calculate_curvature_straight():
    """Test curvature calculation for straight lanes."""
    # Create straight vertical lane polynomials
    left_fit = np.array([0.0, 0.0, 400.0])   # x = 400 (vertical)
    right_fit = np.array([0.0, 0.0, 880.0])  # x = 880 (vertical)

    left_curv, right_curv = calculate_curvature(left_fit, right_fit, 719)

    # Straight lines should have very large curvature radius
    assert left_curv > 10000  # > 10 km (essentially straight)
    assert right_curv > 10000


def test_calculate_curvature_curved():
    """Test curvature calculation for curved lanes."""
    # Create curved lane polynomials
    left_fit = np.array([0.0002, -0.05, 400.0])   # Slightly curved
    right_fit = np.array([0.0002, -0.05, 880.0])

    left_curv, right_curv = calculate_curvature(left_fit, right_fit, 719)

    # Should have reasonable curvature radius
    assert 100 < left_curv < 10000  # Between 100m and 10km
    assert 100 < right_curv < 10000


def test_calculate_vehicle_position_centered():
    """Test vehicle position when centered in lane."""
    # Lanes at x=400 and x=880, center at x=640
    left_fit = np.array([0.0, 0.0, 400.0])
    right_fit = np.array([0.0, 0.0, 880.0])

    offset = calculate_vehicle_position(left_fit, right_fit, 1280)

    # Vehicle center (640) vs lane center (640) -> offset should be ~0
    assert abs(offset) < 0.1  # Within 10cm of center


def test_calculate_vehicle_position_off_center():
    """Test vehicle position when off-center."""
    # Lanes at x=300 and x=780, center at x=540
    # But vehicle center is at x=640
    left_fit = np.array([0.0, 0.0, 300.0])
    right_fit = np.array([0.0, 0.0, 780.0])

    offset = calculate_vehicle_position(left_fit, right_fit, 1280)

    # Vehicle is to the right of lane center
    assert offset > 0.3  # More than 30cm right of center


def test_visualize_lane_detection(binary_warped_with_lanes):
    """Test lane detection visualization."""
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped_with_lanes)
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

    vis_img = visualize_lane_detection(
        binary_warped_with_lanes, left_fit, right_fit,
        leftx, lefty, rightx, righty
    )

    # Check output shape
    assert vis_img.shape == (720, 1280, 3)

    # Check it's a color image
    assert vis_img.dtype == np.uint8

    # Check it has colors (not all zeros)
    assert np.sum(vis_img) > 0


def test_empty_image_handling():
    """Test handling of empty binary image."""
    empty_img = np.zeros((720, 1280), dtype=np.uint8)

    # Should not crash, but may not find many pixels
    leftx, lefty, rightx, righty = find_lane_pixels(empty_img)

    # Arrays should exist (may be empty)
    assert isinstance(leftx, np.ndarray)
    assert isinstance(rightx, np.ndarray)


def test_polynomial_coefficients_format():
    """Test that polynomial coefficients are in correct format."""
    # Create simple test data
    y = np.array([0, 100, 200, 300, 400])
    x = np.array([400, 405, 410, 415, 420])

    fit = np.polyfit(y, x, 2)

    # Should return [A, B, C] for x = Ay^2 + By + C
    assert len(fit) == 3
    assert isinstance(fit, np.ndarray)


def test_lane_width_reasonable():
    """Test that detected lane width is reasonable."""
    img = np.zeros((720, 1280), dtype=np.uint8)
    img[:, 400:404] = 1  # Left lane at x~400
    img[:, 880:884] = 1  # Right lane at x~880

    leftx, lefty, rightx, righty = find_lane_pixels(img)
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

    # Calculate lane width at bottom of image
    y = 719
    left_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    right_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    lane_width_pixels = right_x - left_x

    # Lane width should be ~480 pixels (880-400)
    assert 450 < lane_width_pixels < 510

    # In meters (assuming 3.7m/700px)
    xm_per_pix = 3.7 / 700
    lane_width_meters = lane_width_pixels * xm_per_pix

    # US highway lane is ~3.7m wide
    assert 2.0 < lane_width_meters < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
