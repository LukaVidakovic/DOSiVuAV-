#!/usr/bin/env python3
"""Debug problematičnih slika - ograda umesto linije"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

def load_calibration():
    data = np.load('calibration.npz')
    return data['mtx'], data['dist']

def combined_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
    
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 200) & (l_channel <= 255)] = 1
    
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 90) & (s_channel <= 255)] = 1
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= 155) & (b_channel <= 200)] = 1
    
    combined = np.zeros_like(sobel_binary)
    combined[((sobel_binary == 1) | (l_binary == 1)) & 
             ((sobel_binary == 1) | (s_binary == 1) | (b_binary == 1))] = 1
    
    return combined

def get_perspective_transform_OLD(img_shape):
    """Stari transform - previše širok"""
    h, w = img_shape[:2]
    src = np.float32([
        [w * 0.42, h * 0.65],
        [w * 0.58, h * 0.65],
        [w * 0.10, h * 0.95],  # Previše levo!
        [w * 0.90, h * 0.95]   # Previše desno!
    ])
    dst = np.float32([
        [w * 0.20, 0],
        [w * 0.80, 0],
        [w * 0.20, h],
        [w * 0.80, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src, dst

def get_perspective_transform_NEW(img_shape):
    """Novi transform - uži, fokusiran na traku"""
    h, w = img_shape[:2]
    src = np.float32([
        [w * 0.43, h * 0.65],
        [w * 0.57, h * 0.65],
        [w * 0.20, h * 0.95],  # Manje levo
        [w * 0.80, h * 0.95]   # Manje desno
    ])
    dst = np.float32([
        [w * 0.25, 0],
        [w * 0.75, 0],
        [w * 0.25, h],
        [w * 0.75, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src, dst

def debug_image(img_path):
    print(f"\n{'='*60}")
    print(f"Debug: {img_path}")
    print('='*60)
    
    img = cv2.imread(img_path)
    mtx, dist = load_calibration()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    binary = combined_threshold(undist)
    
    # OLD transform
    M_old, Minv_old, src_old, dst_old = get_perspective_transform_OLD(img.shape)
    warped_old = cv2.warpPerspective(binary, M_old, (img.shape[1], img.shape[0]))
    
    # NEW transform
    M_new, Minv_new, src_new, dst_new = get_perspective_transform_NEW(img.shape)
    warped_new = cv2.warpPerspective(binary, M_new, (img.shape[1], img.shape[0]))
    
    # Draw ROI on original
    img_old = undist.copy()
    pts_old = src_old.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_old, [pts_old], True, (0, 0, 255), 3)
    
    img_new = undist.copy()
    pts_new = src_new.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_new, [pts_new], True, (0, 255, 0), 3)
    
    # Histograms
    hist_old = np.sum(warped_old[warped_old.shape[0]//2:, :], axis=0)
    hist_new = np.sum(warped_new[warped_new.shape[0]//2:, :], axis=0)
    
    midpoint = len(hist_old) // 2
    left_old = np.argmax(hist_old[:midpoint])
    right_old = np.argmax(hist_old[midpoint:]) + midpoint
    left_new = np.argmax(hist_new[:midpoint])
    right_new = np.argmax(hist_new[midpoint:]) + midpoint
    
    print(f"OLD: left={left_old:3d}, right={right_old:4d}, width={right_old-left_old:3d}")
    print(f"NEW: left={left_new:3d}, right={right_new:4d}, width={right_new-left_new:3d}")
    
    # Visualize
    h, w = img.shape[:2]
    
    row1 = np.hstack([cv2.resize(img_old, (w//2, h//2)), 
                      cv2.resize(img_new, (w//2, h//2))])
    
    warped_old_vis = cv2.cvtColor(warped_old*255, cv2.COLOR_GRAY2BGR)
    cv2.line(warped_old_vis, (left_old, 0), (left_old, h), (0, 255, 0), 2)
    cv2.line(warped_old_vis, (right_old, 0), (right_old, h), (0, 0, 255), 2)
    
    warped_new_vis = cv2.cvtColor(warped_new*255, cv2.COLOR_GRAY2BGR)
    cv2.line(warped_new_vis, (left_new, 0), (left_new, h), (0, 255, 0), 2)
    cv2.line(warped_new_vis, (right_new, 0), (right_new, h), (0, 0, 255), 2)
    
    row2 = np.hstack([cv2.resize(warped_old_vis, (w//2, h//2)), 
                      cv2.resize(warped_new_vis, (w//2, h//2))])
    
    grid = np.vstack([row1, row2])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid, "OLD ROI (wide)", (10, 30), font, 0.7, (0,0,255), 2)
    cv2.putText(grid, "NEW ROI (narrow)", (w//2+10, 30), font, 0.7, (0,255,0), 2)
    cv2.putText(grid, "OLD Warped", (10, h//2+30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "NEW Warped", (w//2+10, h//2+30), font, 0.7, (255,255,255), 2)
    
    import os
    basename = os.path.basename(img_path).replace('.jpg', '')
    cv2.imwrite(f'debug/roi_comparison_{basename}.jpg', grid)
    print(f"✓ Saved: debug/roi_comparison_{basename}.jpg")

# Test problematic images
images = [
    'test_images/test1.jpg',
    'test_images/test4.jpg',
    'test_images/challange00101.jpg',
    'test_images/challange00111.jpg',
]

for img_path in images:
    debug_image(img_path)

print(f"\n{'='*60}")
print("ZAKLJUČAK:")
print("OLD ROI zahvata ogradu sa strane")
print("NEW ROI je uži i fokusiran samo na traku")
print("='*60")
