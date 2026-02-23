#!/usr/bin/env python3
"""
Generate Documentation Images

Creates all necessary images showing each pipeline step.
Uses actual functions from lane_detection_final.py to ensure consistency.
"""

import numpy as np
import cv2
import os
import sys

# Import functions from main pipeline
sys.path.insert(0, '.')
from lane_detection_final import (
    load_calibration,
    combined_threshold,
    get_perspective_transform,
    find_lane_pixels,
    fit_polynomial,
    calculate_curvature,
    calculate_vehicle_position,
    draw_lane,
    add_text
)

def generate_pipeline_images(test_image='test_images/test1.jpg'):
    """Generate images for each pipeline step using actual pipeline functions"""
    
    print(f"Generating images for: {test_image}")
    
    # Load calibration
    mtx, dist = load_calibration()
    
    # 1. Original
    img = cv2.imread(test_image)
    cv2.imwrite('docs/01_original.jpg', img)
    print("  ✓ 01_original.jpg")
    
    # 2. Undistorted
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('docs/02_undistorted.jpg', undist)
    print("  ✓ 02_undistorted.jpg")
    
    # 3. Binary threshold (using actual function)
    binary = combined_threshold(undist)
    binary_vis = binary * 255
    cv2.imwrite('docs/03_binary_threshold.jpg', binary_vis)
    print("  ✓ 03_binary_threshold.jpg")
    
    # 4. Perspective transform with source lines (using actual function)
    M, Minv = get_perspective_transform(img.shape)
    
    # Draw source trapezoid
    img_with_lines = undist.copy()
    h, w = img.shape[:2]
    src_points = np.array([
        [w*0.43, h*0.65],
        [w*0.57, h*0.65],
        [w*0.80, h*0.95],
        [w*0.20, h*0.95]
    ], np.int32)
    cv2.polylines(img_with_lines, [src_points], True, (255, 0, 0), 3)
    cv2.imwrite('docs/04_perspective_src.jpg', img_with_lines)
    print("  ✓ 04_perspective_src.jpg")
    
    # 5. Warped binary
    binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
    binary_warped_vis = binary_warped * 255
    cv2.imwrite('docs/05_warped.jpg', binary_warped_vis)
    print("  ✓ 05_warped.jpg")
    
    # 6. Sliding windows (using actual function)
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    
    # Visualize sliding windows
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    nwindows = 9
    window_height = int(binary_warped.shape[0] // nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
                     (win_xleft_high, win_y_high), (0, 255, 0), 3)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), 
                     (win_xright_high, win_y_high), (0, 255, 0), 3)
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    cv2.imwrite('docs/06_sliding_windows.jpg', out_img)
    print("  ✓ 06_sliding_windows.jpg")
    
    # 7. Polynomial fit (using actual function)
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Draw polynomial
    for i in range(len(ploty)):
        cv2.circle(out_img, (int(left_fitx[i]), int(ploty[i])), 2, (255, 255, 0), -1)
        cv2.circle(out_img, (int(right_fitx[i]), int(ploty[i])), 2, (255, 255, 0), -1)
    
    cv2.imwrite('docs/07_polynomial_fit.jpg', out_img)
    print("  ✓ 07_polynomial_fit.jpg")
    
    # 8. Final result (using actual functions)
    result = draw_lane(undist, binary_warped, left_fit, right_fit, Minv)
    
    # Calculate metrics (using actual functions)
    y_eval = binary_warped.shape[0] - 1
    left_curv, right_curv = calculate_curvature(left_fit, right_fit, y_eval, binary_warped.shape)
    avg_curv = (left_curv + right_curv) / 2
    offset = calculate_vehicle_position(left_fit, right_fit, img.shape[1], img.shape[0])
    
    result = add_text(result, avg_curv, offset)
    
    cv2.imwrite('docs/08_final_result.jpg', result)
    print("  ✓ 08_final_result.jpg")

if __name__ == '__main__':
    print("="*60)
    print("GENERATING DOCUMENTATION IMAGES")
    print("="*60)
    
    os.makedirs('docs', exist_ok=True)
    
    generate_pipeline_images('test_images/test1.jpg')
    
    print("\n✓ All images generated in docs/ folder")
    print("  Using actual pipeline functions for consistency")
    print("="*60)
