#!/usr/bin/env python3
"""Debug line overlay alignment"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

from lane_detection_final import (
    load_calibration, combined_threshold, get_perspective_transform,
    find_lane_pixels, fit_polynomial
)

def draw_lane_detailed(undist, binary_warped, left_fit, right_fit, Minv):
    """Draw lane with thinner, more precise lines"""
    
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Green lane area
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    
    # Thinner lines - 15px instead of 25px
    cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), thickness=15)
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

def test_alignment(img_path):
    print(f"\nTesting: {img_path}")
    
    img = cv2.imread(img_path)
    mtx, dist = load_calibration()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    binary = combined_threshold(undist)
    M, Minv = get_perspective_transform(img.shape)
    binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
    
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    
    if len(leftx) < 100 or len(rightx) < 100:
        print("  ✗ Not enough pixels")
        return
    
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
    
    # OLD - thick lines (25px)
    from lane_detection_final import draw_lane
    result_old = draw_lane(undist, binary_warped, left_fit, right_fit, Minv)
    
    # NEW - thin lines (15px)
    result_new = draw_lane_detailed(undist, binary_warped, left_fit, right_fit, Minv)
    
    # Comparison
    h, w = img.shape[:2]
    comparison = np.hstack([cv2.resize(result_old, (w//2, h//2)), 
                           cv2.resize(result_new, (w//2, h//2))])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "OLD (thick 25px)", (10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(comparison, "NEW (thin 15px)", (w//2+10, 30), font, 0.7, (255,255,255), 2)
    
    import os
    basename = os.path.basename(img_path).replace('.jpg', '')
    cv2.imwrite(f'debug/line_thickness_{basename}.jpg', comparison)
    print(f"  ✓ Saved: debug/line_thickness_{basename}.jpg")

# Test on a few images
images = [
    'test_images/straight_lines1.jpg',
    'test_images/test1.jpg',
    'test_images/solidYellowLeft.jpg',
]

print("="*60)
print("TESTIRANJE DEBLJINE LINIJA")
print("="*60)

for img_path in images:
    test_alignment(img_path)

print("\n" + "="*60)
print("Proveri debug/ folder - tanje linije treba bolje da se poklapaju")
print("="*60)
