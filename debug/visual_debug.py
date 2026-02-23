#!/usr/bin/env python3
"""Visual debugging tool for lane detection pipeline"""

import numpy as np
import cv2
import sys
import os

sys.path.insert(0, 'src')

from camera_calibration import load_calibration, undistort_image
from thresholding import combined_threshold, abs_sobel_thresh
from perspective_transform import get_default_src_points, get_default_dst_points, get_perspective_transform, warp_image
from lane_detection import find_lane_pixels, fit_polynomial, visualize_lane_detection

def debug_image(img_path):
    """Debug single image through all pipeline stages"""
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot load: {img_path}")
        return
    
    print(f"\nProcessing: {img_path}")
    print(f"Shape: {img.shape}")
    
    # 1. Undistort
    if os.path.exists("calibration.npz"):
        mtx, dist, _, _ = load_calibration()
        undist = undistort_image(img, mtx, dist)
    else:
        undist = img.copy()
    
    # 2. Thresholding - test different methods
    gradx = abs_sobel_thresh(undist, orient='x', thresh=(20, 100))
    
    # HLS S-channel
    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    hls_s = np.zeros_like(s_channel)
    hls_s[(s_channel >= 90) & (s_channel <= 255)] = 1
    
    # HLS L-channel
    l_channel = hls[:, :, 1]
    hls_l = np.zeros_like(l_channel)
    hls_l[(l_channel >= 200) & (l_channel <= 255)] = 1
    
    combined = combined_threshold(undist)
    
    print(f"Gradx pixels: {np.sum(gradx)}")
    print(f"HLS-S pixels: {np.sum(hls_s)}")
    print(f"HLS-L pixels: {np.sum(hls_l)}")
    print(f"Combined pixels: {np.sum(combined)}")
    
    # 3. Perspective transform
    src = get_default_src_points(img.shape)
    dst = get_default_dst_points(img.shape)
    M, Minv = get_perspective_transform(src, dst)
    binary_warped = warp_image(combined, M)
    
    # 4. Lane detection
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    print(f"Left lane pixels: {len(leftx)}")
    print(f"Right lane pixels: {len(rightx)}")
    
    if len(leftx) > 0 and len(rightx) > 0:
        left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
        lane_vis = visualize_lane_detection(binary_warped, left_fit, right_fit, leftx, lefty, rightx, righty)
    else:
        lane_vis = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # Create visualization grid
    h, w = img.shape[:2]
    
    # Row 1: Original, Undistorted, Combined threshold
    row1_1 = cv2.resize(img, (w//3, h//3))
    row1_2 = cv2.resize(undist, (w//3, h//3))
    row1_3 = cv2.resize(cv2.cvtColor(combined*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row1 = np.hstack([row1_1, row1_2, row1_3])
    
    # Row 2: Gradx, HLS-S, HLS-L
    row2_1 = cv2.resize(cv2.cvtColor(gradx*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row2_2 = cv2.resize(cv2.cvtColor(hls_s*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row2_3 = cv2.resize(cv2.cvtColor(hls_l*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row2 = np.hstack([row2_1, row2_2, row2_3])
    
    # Row 3: Warped binary, Lane detection
    row3_1 = cv2.resize(cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row3_2 = cv2.resize(lane_vis, (w//3, h//3))
    row3_3 = np.zeros((h//3, w//3, 3), dtype=np.uint8)
    row3 = np.hstack([row3_1, row3_2, row3_3])
    
    # Stack all rows
    grid = np.vstack([row1, row2, row3])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid, "Original", (10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "Undistorted", (w//3+10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "Combined", (2*w//3+10, 30), font, 0.7, (255,255,255), 2)
    
    cv2.putText(grid, "Sobel X", (10, h//3+30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "HLS S-channel", (w//3+10, h//3+30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "HLS L-channel", (2*w//3+10, h//3+30), font, 0.7, (255,255,255), 2)
    
    cv2.putText(grid, "Warped", (10, 2*h//3+30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "Lane Detection", (w//3+10, 2*h//3+30), font, 0.7, (255,255,255), 2)
    
    # Save
    basename = os.path.basename(img_path)
    output_path = f"debug/visual_{basename}"
    cv2.imwrite(output_path, grid)
    print(f"Saved: {output_path}\n")

if __name__ == "__main__":
    test_images = [
        "test_images/straight_lines1.jpg",
        "test_images/test1.jpg",
        "test_images/test4.jpg",
        "test_images/challange00136.jpg",
        "test_images/solidWhiteRight.jpg",
        "test_images/solidYellowLeft.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            debug_image(img_path)
