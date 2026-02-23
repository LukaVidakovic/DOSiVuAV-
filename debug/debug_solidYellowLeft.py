#!/usr/bin/env python3
"""
Debug solidYellowLeft.jpg - Zašto ne radi dobro?
"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

from debug.test_faza4_detection import load_calibration, load_perspective_transform, combined_threshold

# Load image
img = cv2.imread('test_images/solidYellowLeft.jpg')
print(f"Image shape: {img.shape}")

# Load calibration
mtx, dist = load_calibration()
undist = cv2.undistort(img, mtx, dist, None, mtx)

# Apply thresholding
binary = combined_threshold(undist)
print(f"Binary pixels: {np.sum(binary)}")

# Load perspective transform (designed for 720p!)
M, Minv = load_perspective_transform()

# Warp
binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
print(f"Warped binary pixels: {np.sum(binary_warped)}")

# Histogram
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
midpoint = len(histogram) // 2
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

print(f"Histogram peaks: left={leftx_base}, right={rightx_base}, width={rightx_base-leftx_base}")

# Visualize
h, w = img.shape[:2]

# Row 1: Original, Binary, Warped Binary
row1_1 = cv2.resize(undist, (w//2, h//2))
row1_2 = cv2.resize(cv2.cvtColor(binary*255, cv2.COLOR_GRAY2BGR), (w//2, h//2))
row1 = np.hstack([row1_1, row1_2])

# Row 2: Warped binary with histogram
warped_vis = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
# Draw histogram peaks
cv2.line(warped_vis, (leftx_base, 0), (leftx_base, h), (0, 255, 0), 3)
cv2.line(warped_vis, (rightx_base, 0), (rightx_base, h), (0, 0, 255), 3)
cv2.line(warped_vis, (midpoint, 0), (midpoint, h), (255, 255, 0), 2)

row2_1 = cv2.resize(warped_vis, (w//2, h//2))
row2_2 = np.zeros((h//2, w//2, 3), dtype=np.uint8)

# Draw histogram
hist_img = np.zeros((h//2, w//2, 3), dtype=np.uint8)
hist_normalized = histogram * (h//2) / np.max(histogram)
for i, val in enumerate(hist_normalized):
    x = int(i * (w//2) / len(histogram))
    cv2.line(hist_img, (x, h//2), (x, h//2 - int(val)), (255, 255, 255), 1)
row2_2 = hist_img

row2 = np.hstack([row2_1, row2_2])

grid = np.vstack([row1, row2])

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(grid, "Original", (10, 30), font, 0.7, (255,255,255), 2)
cv2.putText(grid, "Binary", (w//2+10, 30), font, 0.7, (255,255,255), 2)
cv2.putText(grid, "Warped + Peaks", (10, h//2+30), font, 0.7, (255,255,255), 2)
cv2.putText(grid, "Histogram", (w//2+10, h//2+30), font, 0.7, (255,255,255), 2)

cv2.imwrite('debug/debug_solidYellowLeft.jpg', grid)
print("\n✓ Saved: debug/debug_solidYellowLeft.jpg")

# Problem analysis
print("\n=== ANALIZA PROBLEMA ===")
if binary_warped.shape != (720, 1280):
    print(f"✗ Warped shape je {binary_warped.shape}, a očekivano je (720, 1280)")
    print("  → Perspective transform je dizajniran za 720p slike!")
    print("  → Treba prilagoditi transform za različite rezolucije")

if leftx_base < 50:
    print(f"✗ Leva linija je previše blizu ivice ({leftx_base}px)")
    print("  → Perspective transform ne pokriva dobro levu liniju")

if rightx_base - leftx_base > 900:
    print(f"✗ Širina trake je prevelika ({rightx_base - leftx_base}px)")
    print("  → Perspective transform nije dobro podešen")
