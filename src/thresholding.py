"""
Binary Thresholding Module - Improved for robust lane detection

Combines gradient and color thresholds to detect both white and yellow lanes
in various lighting conditions.
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
    """Apply Sobel edge detection and threshold."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    max_val = np.max(abs_sobel)
    if max_val == 0:
        scaled_sobel = np.zeros_like(abs_sobel, dtype=np.uint8)
    else:
        scaled_sobel = np.uint8(255 * abs_sobel / max_val)

    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary


def combined_threshold(
    img: np.ndarray,
    sobel_kernel: int = 3,
    grad_thresh: Tuple[int, int] = (20, 100),
    hls_s_thresh: Tuple[int, int] = (90, 255),
    hls_l_thresh: Tuple[int, int] = (200, 255),
    lab_b_thresh: Tuple[int, int] = (155, 200)
) -> np.ndarray:
    """
    Combine gradient and color thresholds for robust lane detection.
    
    Uses:
    - Sobel X: detects vertical edges
    - HLS S-channel: detects yellow lines
    - HLS L-channel: detects white lines  
    - LAB B-channel: detects yellow lines (robust)
    """
    # Gradient
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=sobel_kernel, thresh=grad_thresh)

    # HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= hls_s_thresh[0]) & (s_channel <= hls_s_thresh[1])] = 1
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= hls_l_thresh[0]) & (l_channel <= hls_l_thresh[1])] = 1
    
    # LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= lab_b_thresh[0]) & (b_channel <= lab_b_thresh[1])] = 1

    # Combine: (gradient OR white) AND (gradient OR yellow)
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) | (l_binary == 1)) & ((gradx == 1) | (s_binary == 1) | (b_binary == 1))] = 1

    return combined
