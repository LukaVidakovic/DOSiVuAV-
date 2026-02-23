"""
Debug single image - show all pipeline steps
"""
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '../src')

from camera_calibration import load_calibration, undistort_image
from thresholding import combined_threshold
from perspective_transform import get_default_src_points, get_default_dst_points, get_perspective_transform, warp_image
from lane_detection import find_lane_pixels, find_lane_base

# Load calibration
mtx, dist, _, _ = load_calibration('../calibration.npz')

# Load image
img_name = 'solidYellowCurve2'
img = cv2.imread(f'../test_images/{img_name}.jpg')
print(f"Processing: {img_name}")

# Undistort
undist = undistort_image(img, mtx, dist)

# Binary threshold
binary = combined_threshold(undist)
print(f"Binary pixels: {np.sum(binary)} ({100*np.sum(binary)/binary.size:.2f}%)")

# Perspective transform
src = get_default_src_points(img.shape)
dst = get_default_dst_points(img.shape)
M, Minv = get_perspective_transform(src, dst)

binary_warped = warp_image(binary, M)
print(f"Warped binary pixels: {np.sum(binary_warped)} ({100*np.sum(binary_warped)/binary_warped.size:.2f}%)")

# Find lane base
leftx_base, rightx_base = find_lane_base(binary_warped)
print(f"\nLane base positions:")
print(f"  Left: {leftx_base} px")
print(f"  Right: {rightx_base} px")
print(f"  Distance: {rightx_base - leftx_base} px")
print(f"  Midpoint: {(leftx_base + rightx_base) / 2} px")
print(f"  Image center: {binary_warped.shape[1] / 2} px")

# Find lane pixels
leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
print(f"\nDetected pixels:")
print(f"  Left: {len(leftx)} pixels")
print(f"  Right: {len(rightx)} pixels")

# Create visualization with histogram
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Original
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Binary
axes[0, 1].imshow(binary, cmap='gray')
axes[0, 1].set_title('Binary Threshold')
axes[0, 1].axis('off')

# Binary warped
axes[1, 0].imshow(binary_warped, cmap='gray')
axes[1, 0].axvline(x=leftx_base, color='r', linestyle='--', label=f'Left base: {leftx_base}')
axes[1, 0].axvline(x=rightx_base, color='b', linestyle='--', label=f'Right base: {rightx_base}')
axes[1, 0].set_title('Warped Binary + Base Lines')
axes[1, 0].legend()
axes[1, 0].axis('off')

# Histogram
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
axes[1, 1].plot(histogram)
axes[1, 1].axvline(x=leftx_base, color='r', linestyle='--', label=f'Left: {leftx_base}')
axes[1, 1].axvline(x=rightx_base, color='b', linestyle='--', label=f'Right: {rightx_base}')
axes[1, 1].axvline(x=binary_warped.shape[1]//2, color='g', linestyle='--', label='Center')
axes[1, 1].set_title('Histogram (bottom half)')
axes[1, 1].set_xlabel('X position')
axes[1, 1].set_ylabel('Pixel count')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f'../output_images/{img_name}_debug.jpg', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved debug visualization: ../output_images/{img_name}_debug.jpg")
