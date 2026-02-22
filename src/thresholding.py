"""
Binary Thresholding Module

This module provides various thresholding techniques to extract lane features
from road images using gradient and color transformations.

Based on lecture examples: 03_01_sobel.py, 00_04_color_spaces.py
"""

import numpy as np
import cv2
from typing import Tuple


def abs_sobel_thresh(
    img: np.ndarray,
    orient: str = 'x',
    sobel_kernel: int = 3,
    thresh: Tuple[int, int] = (0, 255)
) -> np.ndarray:
    """
    Apply Sobel edge detection and threshold.

    Args:
        img: Input image (grayscale or BGR)
        orient: Gradient orientation ('x' or 'y')
        sobel_kernel: Sobel kernel size (odd number)
        thresh: Threshold range (min, max)

    Returns:
        Binary image where gradient is within threshold
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply Sobel in x or y direction
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take absolute value and scale to 8-bit
    abs_sobel = np.absolute(sobel)

    # Avoid division by zero
    max_val = np.max(abs_sobel)
    if max_val == 0:
        scaled_sobel = np.zeros_like(abs_sobel, dtype=np.uint8)
    else:
        scaled_sobel = np.uint8(255 * abs_sobel / max_val)

    # Apply threshold
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary


def mag_thresh(
    img: np.ndarray,
    sobel_kernel: int = 3,
    thresh: Tuple[int, int] = (0, 255)
) -> np.ndarray:
    """
    Apply magnitude of gradient threshold.

    Args:
        img: Input image (grayscale or BGR)
        sobel_kernel: Sobel kernel size (odd number)
        thresh: Threshold range (min, max)

    Returns:
        Binary image where gradient magnitude is within threshold
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply Sobel in both directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Scale to 8-bit (avoid division by zero)
    max_val = np.max(magnitude)
    if max_val == 0:
        scaled_magnitude = np.zeros_like(magnitude, dtype=np.uint8)
    else:
        scaled_magnitude = np.uint8(255 * magnitude / max_val)

    # Apply threshold
    binary = np.zeros_like(scaled_magnitude)
    binary[(scaled_magnitude >= thresh[0]) & (scaled_magnitude <= thresh[1])] = 1

    return binary


def dir_thresh(
    img: np.ndarray,
    sobel_kernel: int = 3,
    thresh: Tuple[float, float] = (0, np.pi/2)
) -> np.ndarray:
    """
    Apply direction of gradient threshold.

    Args:
        img: Input image (grayscale or BGR)
        sobel_kernel: Sobel kernel size (odd number)
        thresh: Threshold range in radians (min, max)

    Returns:
        Binary image where gradient direction is within threshold
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply Sobel in both directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate direction of gradient
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)

    # Apply threshold
    binary = np.zeros_like(direction, dtype=np.uint8)
    binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return binary


def hls_select(
    img: np.ndarray,
    channel: str = 's',
    thresh: Tuple[int, int] = (0, 255)
) -> np.ndarray:
    """
    Apply HLS color space threshold on specified channel.

    Args:
        img: Input BGR image
        channel: Channel to use ('h', 'l', or 's')
        thresh: Threshold range (min, max)

    Returns:
        Binary image where channel value is within threshold
    """
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Select channel
    channel_map = {'h': 0, 'l': 1, 's': 2}
    channel_idx = channel_map.get(channel.lower(), 2)
    channel_img = hls[:, :, channel_idx]

    # Apply threshold
    binary = np.zeros_like(channel_img)
    binary[(channel_img >= thresh[0]) & (channel_img <= thresh[1])] = 1

    return binary


def rgb_select(
    img: np.ndarray,
    channel: str = 'r',
    thresh: Tuple[int, int] = (0, 255)
) -> np.ndarray:
    """
    Apply RGB color space threshold on specified channel.

    Args:
        img: Input BGR image (OpenCV format)
        channel: Channel to use ('r', 'g', or 'b')
        thresh: Threshold range (min, max)

    Returns:
        Binary image where channel value is within threshold
    """
    # OpenCV uses BGR, so map accordingly
    channel_map = {'b': 0, 'g': 1, 'r': 2}
    channel_idx = channel_map.get(channel.lower(), 2)
    channel_img = img[:, :, channel_idx]

    # Apply threshold
    binary = np.zeros_like(channel_img)
    binary[(channel_img >= thresh[0]) & (channel_img <= thresh[1])] = 1

    return binary


def combined_threshold(
    img: np.ndarray,
    sobel_kernel: int = 3,
    grad_thresh: Tuple[int, int] = (20, 100),
    mag_thresh_val: Tuple[int, int] = (30, 100),
    dir_thresh_val: Tuple[float, float] = (0.7, 1.3),
    hls_thresh: Tuple[int, int] = (170, 255),
    rgb_thresh: Tuple[int, int] = (200, 255)
) -> np.ndarray:
    """
    Combine multiple thresholding techniques to robustly detect lane lines.

    Strategy:
    - Gradient thresholds (Sobel X, magnitude, direction) detect edges
    - Color thresholds (HLS S-channel, RGB R-channel) detect yellow/white lanes

    Args:
        img: Input BGR image
        sobel_kernel: Sobel kernel size
        grad_thresh: Sobel X gradient threshold
        mag_thresh_val: Magnitude threshold
        dir_thresh_val: Direction threshold (radians)
        hls_thresh: HLS S-channel threshold (good for yellow/white)
        rgb_thresh: RGB R-channel threshold (good for yellow)

    Returns:
        Combined binary image
    """
    # Gradient thresholds
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=sobel_kernel, thresh=grad_thresh)
    mag = mag_thresh(img, sobel_kernel=sobel_kernel, thresh=mag_thresh_val)
    direction = dir_thresh(img, sobel_kernel=sobel_kernel, thresh=dir_thresh_val)

    # Combine gradient thresholds
    gradient_binary = np.zeros_like(gradx)
    gradient_binary[((gradx == 1) & (mag == 1)) & (direction == 1)] = 1

    # Color thresholds
    hls_binary = hls_select(img, channel='s', thresh=hls_thresh)
    rgb_binary = rgb_select(img, channel='r', thresh=rgb_thresh)

    # Combine color thresholds
    color_binary = np.zeros_like(hls_binary)
    color_binary[(hls_binary == 1) | (rgb_binary == 1)] = 1

    # Combine gradient and color
    combined = np.zeros_like(gradient_binary)
    combined[(gradient_binary == 1) | (color_binary == 1)] = 1

    return combined


def region_of_interest(
    img: np.ndarray,
    vertices: np.ndarray
) -> np.ndarray:
    """
    Apply region of interest mask to focus on road area.

    Args:
        img: Input binary image
        vertices: Polygon vertices defining region of interest

    Returns:
        Masked binary image
    """
    # Create blank mask
    mask = np.zeros_like(img)

    # Fill polygon with white
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Apply mask
    masked = cv2.bitwise_and(img, mask)

    return masked


if __name__ == "__main__":
    """
    Example usage: Test thresholding on a test image
    """
    print("=" * 60)
    print("Binary Thresholding Test")
    print("=" * 60)

    # Load test image
    img = cv2.imread("test_images/test1.jpg")

    if img is None:
        print("✗ Could not load test image")
    else:
        print("✓ Loaded test image")

        # Apply combined threshold
        binary = combined_threshold(img)

        # Create visualization
        # Stack original (as grayscale) and binary side by side
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_viz = binary * 255  # Scale to 255 for visualization

        comparison = np.hstack((gray, binary_viz))
        cv2.imwrite("docs/binary_threshold_comparison.jpg", comparison)
        print("✓ Saved binary threshold comparison to: docs/binary_threshold_comparison.jpg")

        # Test individual thresholds
        print("\nTesting individual thresholds:")

        gradx = abs_sobel_thresh(img, orient='x', thresh=(20, 100))
        print(f"  Sobel X: {np.sum(gradx)} pixels activated")

        hls_s = hls_select(img, channel='s', thresh=(170, 255))
        print(f"  HLS S-channel: {np.sum(hls_s)} pixels activated")

        combined = combined_threshold(img)
        print(f"  Combined: {np.sum(combined)} pixels activated")

        print("\n✓ Thresholding test complete!")
