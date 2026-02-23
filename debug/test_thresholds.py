"""
Test individual thresholds to see what produces noise
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '../src')

from camera_calibration import load_calibration, undistort_image
from thresholding import abs_sobel_thresh, mag_thresh, dir_thresh, hls_select, rgb_select

# Load calibration
mtx, dist, _, _ = load_calibration('../calibration.npz')

# Load test image
img = cv2.imread('../test_images/test1.jpg')
undist = undistort_image(img, mtx, dist)

# Test individual thresholds
print("Testing individual thresholds on test1.jpg:")
print("="*60)

# Gradient thresholds
gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=3, thresh=(20, 100))
print(f"Sobel X (20-100): {np.sum(gradx)} pixels ({100*np.sum(gradx)/gradx.size:.2f}%)")

mag = mag_thresh(undist, sobel_kernel=3, thresh=(30, 100))
print(f"Magnitude (30-100): {np.sum(mag)} pixels ({100*np.sum(mag)/mag.size:.2f}%)")

direction = dir_thresh(undist, sobel_kernel=3, thresh=(0.7, 1.3))
print(f"Direction (0.7-1.3): {np.sum(direction)} pixels ({100*np.sum(direction)/direction.size:.2f}%)")

# Combine gradient thresholds (must satisfy ALL three)
gradient_binary = np.zeros_like(gradx)
gradient_binary[((gradx == 1) & (mag == 1)) & (direction == 1)] = 1
print(f"Gradient combined (ALL 3): {np.sum(gradient_binary)} pixels ({100*np.sum(gradient_binary)/gradient_binary.size:.2f}%)")

# Color thresholds
hls_binary = hls_select(undist, channel='s', thresh=(170, 255))
print(f"\nHLS S-channel (170-255): {np.sum(hls_binary)} pixels ({100*np.sum(hls_binary)/hls_binary.size:.2f}%)")

rgb_binary = rgb_select(undist, channel='r', thresh=(200, 255))
print(f"RGB R-channel (200-255): {np.sum(rgb_binary)} pixels ({100*np.sum(rgb_binary)/rgb_binary.size:.2f}%)")

# Combine color thresholds
color_binary = np.zeros_like(hls_binary)
color_binary[(hls_binary == 1) | (rgb_binary == 1)] = 1
print(f"Color combined (OR): {np.sum(color_binary)} pixels ({100*np.sum(color_binary)/color_binary.size:.2f}%)")

# Final combined
combined = np.zeros_like(gradient_binary)
combined[(gradient_binary == 1) | (color_binary == 1)] = 1
print(f"\nFinal combined: {np.sum(combined)} pixels ({100*np.sum(combined)/combined.size:.2f}%)")

# Save visualizations
cv2.imwrite('../docs/debug_gradx.jpg', gradx * 255)
cv2.imwrite('../docs/debug_gradient_combined.jpg', gradient_binary * 255)
cv2.imwrite('../docs/debug_hls.jpg', hls_binary * 255)
cv2.imwrite('../docs/debug_rgb.jpg', rgb_binary * 255)
cv2.imwrite('../docs/debug_color_combined.jpg', color_binary * 255)
cv2.imwrite('../docs/debug_final.jpg', combined * 255)

print("\n✓ Saved debug images to ../docs/debug_*.jpg")
