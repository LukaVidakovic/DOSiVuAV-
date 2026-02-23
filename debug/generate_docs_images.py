"""
Generate documentation images showing pipeline steps
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '../src')

from camera_calibration import load_calibration, undistort_image
from thresholding import combined_threshold
from perspective_transform import get_default_src_points, get_default_dst_points, get_perspective_transform, warp_image
from lane_detection import find_lane_pixels, fit_polynomial

# Load calibration
mtx, dist, _, _ = load_calibration('../calibration.npz')

# Load test image
img = cv2.imread('../test_images/test1.jpg')
undist = undistort_image(img, mtx, dist)

# 1. Binary combo example (grayscale vs binary)
binary = combined_threshold(undist)
gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
binary_viz = binary * 255
comparison = np.hstack((gray, binary_viz))
cv2.imwrite('../docs/binary_combo_example.jpg', comparison)
print("✓ Saved binary_combo_example.jpg")

# 2. Warped straight lines (perspective transform)
src = get_default_src_points(img.shape)
dst = get_default_dst_points(img.shape)
M, Minv = get_perspective_transform(src, dst)

# Draw source points on original
undist_copy = undist.copy()
pts = src.astype(np.int32)
cv2.polylines(undist_copy, [pts], True, (255, 0, 0), 3)

# Warp image
warped = warp_image(undist, M)
warped_copy = warped.copy()
pts_dst = dst.astype(np.int32)
cv2.polylines(warped_copy, [pts_dst], True, (255, 0, 0), 3)

comparison = np.hstack((undist_copy, warped_copy))
cv2.imwrite('../docs/warped_straight_lines.jpg', comparison)
print("✓ Saved warped_straight_lines.jpg")

# 3. Color fit lines (detected pixels + fitted polynomials)
binary_warped = warp_image(binary, M)
leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

if len(leftx) > 0 and len(rightx) > 0:
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

    # Create color visualization
    out_img = np.dstack((binary_warped*255, binary_warped*255, binary_warped*255))

    # Color detected pixels
    out_img[lefty, leftx] = [255, 0, 0]  # Red for left lane
    out_img[righty, rightx] = [0, 0, 255]  # Blue for right lane

    # Generate y values and calculate x values
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Draw polynomials
    for i in range(len(ploty)):
        cv2.circle(out_img, (int(left_fitx[i]), int(ploty[i])), 2, (0, 255, 0), -1)
        cv2.circle(out_img, (int(right_fitx[i]), int(ploty[i])), 2, (0, 255, 0), -1)

    cv2.imwrite('../docs/color_fit_lines.jpg', out_img)
    print("✓ Saved color_fit_lines.jpg")

# 4. Example output (copy final result)
import shutil
shutil.copy('../test_images/test1_output.jpg', '../docs/example_output.jpg')
print("✓ Saved example_output.jpg")

# 5. Undistort comparison (already exists, but regenerate)
test_img = cv2.imread('../test_images/test1.jpg')
undistorted = undistort_image(test_img, mtx, dist)
comparison = np.hstack((test_img, undistorted))
cv2.imwrite('../docs/undistort_comparison.jpg', comparison)
print("✓ Saved undistort_comparison.jpg")

print("\n✓ All documentation images generated!")
