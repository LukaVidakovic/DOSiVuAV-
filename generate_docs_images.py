#!/usr/bin/env python3
"""
Generate Documentation Images

Creates all necessary images showing each pipeline step.
"""

import numpy as np
import cv2
import os

# Load calibration
data = np.load('calibration.npz')
mtx, dist = data['mtx'], data['dist']

# Constants
YM_PER_PIX = 30 / 720
XM_PER_PIX = 3.7 / 700

def combined_threshold(img):
    """HSV color space with morphological closing"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 160), (40, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 200), (255, 20, 255))
    combined = cv2.bitwise_or(yellow, white)
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    return combined // 255

def get_perspective_transform(img_shape):
    """Calculate perspective transform matrices"""
    h, w = img_shape[:2]
    
    src = np.float32([
        [w*0.43, h*0.65],
        [w*0.57, h*0.65],
        [w*0.20, h*0.95],
        [w*0.80, h*0.95]
    ])
    
    dst = np.float32([
        [w*0.25, 0],
        [w*0.75, 0],
        [w*0.25, h],
        [w*0.75, h]
    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return M, Minv, src

def generate_pipeline_images(test_image='test_images/test1.jpg'):
    """Generate images for each pipeline step"""
    
    print(f"Generating images for: {test_image}")
    
    # 1. Original
    img = cv2.imread(test_image)
    cv2.imwrite('docs/01_original.jpg', img)
    print("  ✓ 01_original.jpg")
    
    # 2. Undistorted
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('docs/02_undistorted.jpg', undist)
    print("  ✓ 02_undistorted.jpg")
    
    # 3. Binary threshold
    binary = combined_threshold(undist)
    binary_vis = binary * 255
    cv2.imwrite('docs/03_binary_threshold.jpg', binary_vis)
    print("  ✓ 03_binary_threshold.jpg")
    
    # 4. Perspective transform with source lines
    M, Minv, src_points = get_perspective_transform(img.shape)
    img_with_lines = undist.copy()
    pts = src_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_with_lines, [pts], True, (255, 0, 0), 3)
    cv2.imwrite('docs/04_perspective_src.jpg', img_with_lines)
    print("  ✓ 04_perspective_src.jpg")
    
    # 5. Warped binary
    binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
    binary_warped_vis = binary_warped * 255
    cv2.imwrite('docs/05_warped.jpg', binary_warped_vis)
    print("  ✓ 05_warped.jpg")
    
    # 6. Sliding windows
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
    
    left_lane_inds = []
    right_lane_inds = []
    
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
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    cv2.imwrite('docs/06_sliding_windows.jpg', out_img)
    print("  ✓ 06_sliding_windows.jpg")
    
    # 7. Polynomial fit
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Draw polynomial
    for i in range(len(ploty)):
        cv2.circle(out_img, (int(left_fitx[i]), int(ploty[i])), 2, (255, 255, 0), -1)
        cv2.circle(out_img, (int(right_fitx[i]), int(ploty[i])), 2, (255, 255, 0), -1)
    
    cv2.imwrite('docs/07_polynomial_fit.jpg', out_img)
    print("  ✓ 07_polynomial_fit.jpg")
    
    # 8. Final result
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    right_line = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
    cv2.polylines(color_warp, left_line, False, (255, 0, 0), 15)
    cv2.polylines(color_warp, right_line, False, (0, 0, 255), 15)
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    cv2.imwrite('docs/08_final_result.jpg', result)
    print("  ✓ 08_final_result.jpg")

if __name__ == '__main__':
    print("="*60)
    print("GENERATING DOCUMENTATION IMAGES")
    print("="*60)
    
    os.makedirs('docs', exist_ok=True)
    
    generate_pipeline_images('test_images/test1.jpg')
    
    print("\n✓ All images generated in docs/ folder")
    print("="*60)
