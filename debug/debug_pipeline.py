"""
Debug pipeline - visualize all intermediate steps
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '../src')

from camera_calibration import load_calibration, undistort_image
from thresholding import combined_threshold, enhance_image, adaptive_yellow_threshold
from perspective_transform import get_default_src_points, get_default_dst_points, get_perspective_transform, warp_image
from lane_detection import find_lane_pixels, fit_polynomial

# Load calibration
mtx, dist, _, _ = load_calibration('../calibration.npz')

# Test image
img = cv2.imread('../test_images/challange00101.jpg')
print(f"Original image shape: {img.shape}")

# Undistort
undist = undistort_image(img, mtx, dist)

# Test WITH and WITHOUT CLAHE
print("\n=== Testing WITHOUT CLAHE ===")
binary_no_clahe = combined_threshold(undist, use_clahe=False)
print(f"Pixels detected (no CLAHE): {np.sum(binary_no_clahe)} ({100*np.sum(binary_no_clahe)/(binary_no_clahe.shape[0]*binary_no_clahe.shape[1]):.2f}%)")

print("\n=== Testing WITH CLAHE ===")
binary_with_clahe = combined_threshold(undist, use_clahe=True)
print(f"Pixels detected (with CLAHE): {np.sum(binary_with_clahe)} ({100*np.sum(binary_with_clahe)/(binary_with_clahe.shape[0]*binary_with_clahe.shape[1]):.2f}%)")

# Perspective transform
src = get_default_src_points(img.shape)
dst = get_default_dst_points(img.shape)
M, Minv = get_perspective_transform(src, dst)

binary_warped_no = warp_image(binary_no_clahe, M)
binary_warped_yes = warp_image(binary_with_clahe, M)

# Find pixels
print("\n=== Lane pixel detection (NO CLAHE) ===")
leftx_no, lefty_no, rightx_no, righty_no = find_lane_pixels(binary_warped_no)
print(f"Left pixels: {len(leftx_no)}, Right pixels: {len(rightx_no)}")

if len(leftx_no) > 50 and len(rightx_no) > 50:
    left_fit_no, right_fit_no = fit_polynomial(leftx_no, lefty_no, rightx_no, righty_no, use_robust=True)
    print(f"Left fit: A={left_fit_no[0]:.6f}, B={left_fit_no[1]:.6f}, C={left_fit_no[2]:.2f}")
    print(f"Right fit: A={right_fit_no[0]:.6f}, B={right_fit_no[1]:.6f}, C={right_fit_no[2]:.2f}")
    print(f"Left curve direction: {'RIGHT' if left_fit_no[0] > 0 else 'LEFT'}")
    print(f"Right curve direction: {'RIGHT' if right_fit_no[0] > 0 else 'LEFT'}")

print("\n=== Lane pixel detection (WITH CLAHE) ===")
leftx_yes, lefty_yes, rightx_yes, righty_yes = find_lane_pixels(binary_warped_yes)
print(f"Left pixels: {len(leftx_yes)}, Right pixels: {len(rightx_yes)}")

if len(leftx_yes) > 50 and len(rightx_yes) > 50:
    left_fit_yes, right_fit_yes = fit_polynomial(leftx_yes, lefty_yes, rightx_yes, righty_yes, use_robust=True)
    print(f"Left fit: A={left_fit_yes[0]:.6f}, B={left_fit_yes[1]:.6f}, C={left_fit_yes[2]:.2f}")
    print(f"Right fit: A={right_fit_yes[0]:.6f}, B={right_fit_yes[1]:.6f}, C={right_fit_yes[2]:.2f}")
    print(f"Left curve direction: {'RIGHT' if left_fit_yes[0] > 0 else 'LEFT'}")
    print(f"Right curve direction: {'RIGHT' if right_fit_yes[0] > 0 else 'LEFT'}")

# Save comparison images
cv2.imwrite('debug_binary_no_clahe.jpg', binary_no_clahe * 255)
cv2.imwrite('debug_binary_with_clahe.jpg', binary_with_clahe * 255)
cv2.imwrite('debug_warped_no_clahe.jpg', binary_warped_no * 255)
cv2.imwrite('debug_warped_with_clahe.jpg', binary_warped_yes * 255)

print("\n✓ Saved debug images")
