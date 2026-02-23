#!/usr/bin/env python3
"""Improved lane detection with outlier rejection"""

import numpy as np
import cv2

def find_lane_pixels_improved(binary_warped, nwindows=9, margin=100, minpix=50):
    """Improved sliding window with better centering"""
    
    # Histogram
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    window_height = binary_warped.shape[0] // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        # Narrower margin for better precision
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Recenter using median instead of mean (more robust to outliers)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.median(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.median(nonzerox[good_right_inds]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty

def fit_polynomial_robust(leftx, lefty, rightx, righty, max_residual=50):
    """Fit polynomial with outlier rejection"""
    
    # Initial fit
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Calculate residuals
    left_pred = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
    right_pred = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
    
    left_residuals = np.abs(leftx - left_pred)
    right_residuals = np.abs(rightx - right_pred)
    
    # Remove outliers
    left_inliers = left_residuals < max_residual
    right_inliers = right_residuals < max_residual
    
    # Refit without outliers
    if np.sum(left_inliers) > 100:
        left_fit = np.polyfit(lefty[left_inliers], leftx[left_inliers], 2)
    
    if np.sum(right_inliers) > 100:
        right_fit = np.polyfit(righty[right_inliers], rightx[right_inliers], 2)
    
    return left_fit, right_fit

# Test
import sys
sys.path.insert(0, '.')
from lane_detection_final import load_calibration, combined_threshold, get_perspective_transform

def test_improved(img_path):
    print(f"\n{'='*60}")
    print(f"Testing: {img_path}")
    print('='*60)
    
    img = cv2.imread(img_path)
    mtx, dist = load_calibration()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    binary = combined_threshold(undist)
    M, Minv = get_perspective_transform(img.shape)
    binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
    
    # OLD method
    from lane_detection_final import find_lane_pixels, fit_polynomial
    leftx_old, lefty_old, rightx_old, righty_old = find_lane_pixels(binary_warped)
    left_fit_old, right_fit_old = fit_polynomial(leftx_old, lefty_old, rightx_old, righty_old)
    
    # NEW method
    leftx_new, lefty_new, rightx_new, righty_new = find_lane_pixels_improved(binary_warped)
    left_fit_new, right_fit_new = fit_polynomial_robust(leftx_new, lefty_new, rightx_new, righty_new)
    
    # Compare residuals
    left_pred_old = left_fit_old[0]*lefty_old**2 + left_fit_old[1]*lefty_old + left_fit_old[2]
    left_residuals_old = np.abs(leftx_old - left_pred_old)
    
    left_pred_new = left_fit_new[0]*lefty_new**2 + left_fit_new[1]*lefty_new + left_fit_new[2]
    left_residuals_new = np.abs(leftx_new - left_pred_new)
    
    print(f"OLD: Left residual mean={np.mean(left_residuals_old):.2f}px, max={np.max(left_residuals_old):.2f}px")
    print(f"NEW: Left residual mean={np.mean(left_residuals_new):.2f}px, max={np.max(left_residuals_new):.2f}px")
    
    # Visualize
    warped_vis = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
    warped_vis[lefty_new, leftx_new] = [255, 0, 0]
    warped_vis[righty_new, rightx_new] = [0, 0, 255]
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit_new[0]*ploty**2 + left_fit_new[1]*ploty + left_fit_new[2]
    right_fitx = right_fit_new[0]*ploty**2 + right_fit_new[1]*ploty + right_fit_new[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    cv2.polylines(warped_vis, np.int32([pts_left]), False, (0, 255, 255), thickness=3)
    cv2.polylines(warped_vis, np.int32([pts_right]), False, (0, 255, 255), thickness=3)
    
    import os
    basename = os.path.basename(img_path).replace('.jpg', '')
    cv2.imwrite(f'debug/improved_fit_{basename}.jpg', warped_vis)
    print(f"✓ Saved: debug/improved_fit_{basename}.jpg")

images = [
    'test_images/test1.jpg',
    'test_images/test4.jpg',
    'test_images/challange00101.jpg',
]

for img_path in images:
    test_improved(img_path)

print("\n" + "="*60)
print("Improved method uses median centering and outlier rejection")
print("="*60)
