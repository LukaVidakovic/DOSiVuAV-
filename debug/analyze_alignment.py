#!/usr/bin/env python3
"""Detaljni debug preklapanja linija"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

from lane_detection_final import (
    load_calibration, combined_threshold, get_perspective_transform,
    find_lane_pixels, fit_polynomial
)

def visualize_detailed(img_path):
    print(f"\n{'='*60}")
    print(f"Analyzing: {img_path}")
    print('='*60)
    
    img = cv2.imread(img_path)
    mtx, dist = load_calibration()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Binary
    binary = combined_threshold(undist)
    
    # Perspective
    M, Minv = get_perspective_transform(img.shape)
    binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
    
    # Find pixels
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    
    if len(leftx) < 100 or len(rightx) < 100:
        print("Not enough pixels")
        return
    
    # Fit
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
    
    # Generate fitted line
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Visualize on warped
    warped_vis = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
    
    # Draw detected pixels
    warped_vis[lefty, leftx] = [255, 0, 0]  # Red
    warped_vis[righty, rightx] = [0, 0, 255]  # Blue
    
    # Draw fitted lines
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    cv2.polylines(warped_vis, np.int32([pts_left]), False, (0, 255, 255), thickness=3)
    cv2.polylines(warped_vis, np.int32([pts_right]), False, (0, 255, 255), thickness=3)
    
    # Unwarp fitted lines to original
    line_warp = np.zeros_like(binary_warped).astype(np.uint8)
    line_warp = np.dstack((line_warp, line_warp, line_warp))
    cv2.polylines(line_warp, np.int32([pts_left]), False, (255, 0, 0), thickness=15)
    cv2.polylines(line_warp, np.int32([pts_right]), False, (0, 0, 255), thickness=15)
    
    line_unwarp = cv2.warpPerspective(line_warp, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(undist, 1, line_unwarp, 0.7, 0)
    
    # Create grid
    h, w = img.shape[:2]
    
    row1 = np.hstack([cv2.resize(undist, (w//2, h//2)), 
                      cv2.resize(result, (w//2, h//2))])
    row2 = np.hstack([cv2.resize(cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR), (w//2, h//2)),
                      cv2.resize(warped_vis, (w//2, h//2))])
    
    grid = np.vstack([row1, row2])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid, "Original", (10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "With Overlay", (w//2+10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "Warped Binary", (10, h//2+30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "Detected Pixels + Fit", (w//2+10, h//2+30), font, 0.7, (255,255,255), 2)
    
    import os
    basename = os.path.basename(img_path).replace('.jpg', '')
    cv2.imwrite(f'debug/alignment_analysis_{basename}.jpg', grid)
    print(f"✓ Saved: debug/alignment_analysis_{basename}.jpg")
    
    # Check fit quality
    left_residuals = np.abs(leftx - (left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]))
    right_residuals = np.abs(rightx - (right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]))
    
    print(f"Left fit residual: mean={np.mean(left_residuals):.2f}px, max={np.max(left_residuals):.2f}px")
    print(f"Right fit residual: mean={np.mean(right_residuals):.2f}px, max={np.max(right_residuals):.2f}px")
    
    # Check lane width consistency
    lane_widths = right_fitx - left_fitx
    print(f"Lane width: mean={np.mean(lane_widths):.0f}px, std={np.std(lane_widths):.0f}px")
    print(f"  Bottom: {lane_widths[-1]:.0f}px, Top: {lane_widths[0]:.0f}px")

# Test problematic images
images = [
    'test_images/test1.jpg',
    'test_images/test4.jpg',
    'test_images/challange00101.jpg',
    'test_images/challange00111.jpg',
]

for img_path in images:
    visualize_detailed(img_path)

print("\n" + "="*60)
print("Proveri debug/ folder za detaljnu analizu")
print("="*60)
