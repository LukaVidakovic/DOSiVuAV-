"""
Lane Detection Module

This module provides functions to detect lane lines in bird's-eye view images,
fit polynomials, and calculate lane curvature and vehicle position.

Uses sliding window search and polynomial fitting techniques.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List


def find_lane_base(
    binary_warped: np.ndarray,
    visualize: bool = False
) -> Tuple[int, int]:
    """
    Find the starting x-positions of left and right lanes using histogram.

    Args:
        binary_warped: Binary bird's-eye view image
        visualize: Whether to return histogram (for debugging)

    Returns:
        leftx_base: X-position of left lane base
        rightx_base: X-position of right lane base
    """
    # Take histogram of bottom half of image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    # Find peaks in left and right halves
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    if visualize:
        return leftx_base, rightx_base, histogram

    return leftx_base, rightx_base


def find_lane_pixels(
    binary_warped: np.ndarray,
    nwindows: int = 9,
    margin: int = 100,
    minpix: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find lane pixels using sliding window search.

    Args:
        binary_warped: Binary bird's-eye view image
        nwindows: Number of sliding windows
        margin: Width of windows (+/- margin)
        minpix: Minimum pixels to recenter window

    Returns:
        leftx: X coordinates of left lane pixels
        lefty: Y coordinates of left lane pixels
        rightx: X coordinates of right lane pixels
        righty: Y coordinates of right lane pixels
    """
    # Find lane base positions
    leftx_base, rightx_base = find_lane_base(binary_warped)

    # Set height of windows
    window_height = binary_warped.shape[0] // nwindows

    # Identify x and y positions of all nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions (will be updated for each window)
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Lists to store lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through windows
    for window in range(nwindows):
        # Identify window boundaries
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify nonzero pixels in window
        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]

        good_right_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]

        # Append indices to lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter windows if enough pixels found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(
    leftx: np.ndarray,
    lefty: np.ndarray,
    rightx: np.ndarray,
    righty: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit second order polynomial to lane pixels.

    Args:
        leftx: X coordinates of left lane pixels
        lefty: Y coordinates of left lane pixels
        rightx: X coordinates of right lane pixels
        righty: Y coordinates of right lane pixels

    Returns:
        left_fit: Polynomial coefficients for left lane [A, B, C] where x = Ay^2 + By + C
        right_fit: Polynomial coefficients for right lane [A, B, C]
    """
    # Fit polynomial: x = A*y^2 + B*y + C
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def search_around_poly(
    binary_warped: np.ndarray,
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    margin: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Search for lane pixels around previous polynomial (optimization for video).

    Args:
        binary_warped: Binary bird's-eye view image
        left_fit: Previous left polynomial coefficients
        right_fit: Previous right polynomial coefficients
        margin: Width of search area around polynomial

    Returns:
        leftx, lefty, rightx, righty: Lane pixel coordinates
    """
    # Identify nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Calculate polynomial values for all y positions
    left_fitx = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]
    right_fitx = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]

    # Find pixels within margin of polynomial
    left_lane_inds = ((nonzerox > (left_fitx - margin)) &
                      (nonzerox < (left_fitx + margin)))
    right_lane_inds = ((nonzerox > (right_fitx - margin)) &
                       (nonzerox < (right_fitx + margin)))

    # Extract pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def calculate_curvature(
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    y_eval: int,
    ym_per_pix: float = 30/720,
    xm_per_pix: float = 3.7/700
) -> Tuple[float, float]:
    """
    Calculate radius of curvature for left and right lanes in meters.

    Args:
        left_fit: Left polynomial coefficients (in pixels)
        right_fit: Right polynomial coefficients (in pixels)
        y_eval: Y-value where to evaluate curvature (typically image bottom)
        ym_per_pix: Meters per pixel in y dimension
        xm_per_pix: Meters per pixel in x dimension

    Returns:
        left_curverad: Left lane curvature radius (meters)
        right_curverad: Right lane curvature radius (meters)
    """
    # Generate y values for fitting in world space
    ploty = np.linspace(0, y_eval, num=y_eval+1)

    # Calculate x values using polynomial
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Fit polynomial in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate curvature radius at y_eval
    y_eval_world = y_eval * ym_per_pix

    # R = (1 + (2*A*y + B)^2)^(3/2) / |2*A|
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_world + left_fit_cr[1])**2)**1.5) / \
                    np.abs(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_world + right_fit_cr[1])**2)**1.5) / \
                     np.abs(2*right_fit_cr[0])

    return left_curverad, right_curverad


def calculate_vehicle_position(
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    img_width: int,
    xm_per_pix: float = 3.7/700
) -> float:
    """
    Calculate vehicle position relative to lane center.

    Args:
        left_fit: Left polynomial coefficients
        right_fit: Right polynomial coefficients
        img_width: Image width in pixels
        xm_per_pix: Meters per pixel in x dimension

    Returns:
        Vehicle offset from center (meters). Negative = left, positive = right
    """
    # Calculate lane positions at bottom of image
    y_max = 720  # Assume 720p image

    left_lane_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    right_lane_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]

    # Calculate lane center
    lane_center = (left_lane_pos + right_lane_pos) / 2

    # Calculate vehicle center (assume camera is centered)
    vehicle_center = img_width / 2

    # Calculate offset in meters
    offset = (vehicle_center - lane_center) * xm_per_pix

    return offset


def visualize_lane_detection(
    binary_warped: np.ndarray,
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    leftx: np.ndarray,
    lefty: np.ndarray,
    rightx: np.ndarray,
    righty: np.ndarray
) -> np.ndarray:
    """
    Create visualization of lane detection with polynomial fit.

    Args:
        binary_warped: Binary bird's-eye view image
        left_fit: Left polynomial coefficients
        right_fit: Right polynomial coefficients
        leftx, lefty, rightx, righty: Lane pixel coordinates

    Returns:
        Visualization image
    """
    # Create output image
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Generate y values
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    # Calculate x values from polynomial
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Color lane pixels
    out_img[lefty, leftx] = [255, 0, 0]  # Red for left lane
    out_img[righty, rightx] = [0, 0, 255]  # Blue for right lane

    # Plot polynomial lines
    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])

    cv2.polylines(out_img, np.int32([left_line_pts]), isClosed=False,
                  color=(255, 255, 0), thickness=5)
    cv2.polylines(out_img, np.int32([right_line_pts]), isClosed=False,
                  color=(255, 255, 0), thickness=5)

    return out_img


if __name__ == "__main__":
    """
    Example usage: Test lane detection on a test image
    """
    print("=" * 60)
    print("Lane Detection Test")
    print("=" * 60)

    # Load test image and apply preprocessing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from camera_calibration import load_calibration, undistort_image
    from thresholding import combined_threshold
    from perspective_transform import get_default_src_points, get_default_dst_points, warp_image, get_perspective_transform

    # Load and process image
    img = cv2.imread("test_images/test1.jpg")

    if img is None:
        print("✗ Could not load test image")
    else:
        print("✓ Loaded test image")

        # Apply camera calibration if available
        if os.path.exists("calibration.npz"):
            mtx, dist, _, _ = load_calibration()
            img = undistort_image(img, mtx, dist)
            print("✓ Applied camera calibration")

        # Apply thresholding
        binary = combined_threshold(img)
        print("✓ Applied binary thresholding")

        # Apply perspective transform
        src = get_default_src_points(img.shape)
        dst = get_default_dst_points(img.shape)
        M, Minv = get_perspective_transform(src, dst)
        binary_warped = warp_image(binary, M)
        print("✓ Applied perspective transform")

        # Find lane pixels
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
        print(f"✓ Found {len(leftx)} left lane pixels, {len(rightx)} right lane pixels")

        # Fit polynomial
        left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
        print("✓ Fitted polynomials")
        print(f"  Left:  x = {left_fit[0]:.2e}*y^2 + {left_fit[1]:.2e}*y + {left_fit[2]:.2f}")
        print(f"  Right: x = {right_fit[0]:.2e}*y^2 + {right_fit[1]:.2e}*y + {right_fit[2]:.2f}")

        # Calculate curvature
        left_curv, right_curv = calculate_curvature(left_fit, right_fit, img.shape[0]-1)
        print(f"✓ Calculated curvature:")
        print(f"  Left:  {left_curv:.0f} m")
        print(f"  Right: {right_curv:.0f} m")
        print(f"  Average: {(left_curv + right_curv)/2:.0f} m")

        # Calculate vehicle position
        offset = calculate_vehicle_position(left_fit, right_fit, img.shape[1])
        print(f"✓ Vehicle position: {offset:.2f} m from center")
        if offset < 0:
            print(f"  (Vehicle is {abs(offset):.2f} m left of center)")
        else:
            print(f"  (Vehicle is {offset:.2f} m right of center)")

        # Visualize
        vis_img = visualize_lane_detection(binary_warped, left_fit, right_fit,
                                           leftx, lefty, rightx, righty)
        cv2.imwrite("docs/lane_detection_visualization.jpg", vis_img)
        print("\n✓ Saved visualization to: docs/lane_detection_visualization.jpg")

        print("\n✓ Lane detection test complete!")
