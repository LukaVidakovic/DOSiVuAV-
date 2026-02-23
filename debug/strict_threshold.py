#!/usr/bin/env python3
"""Stricter threshold for cleaner lane detection"""

import numpy as np
import cv2

def strict_threshold(img):
    """Stricter thresholding - less noise, cleaner lines"""
    
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    
    # White - very strict
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 230) & (l_channel <= 255)] = 1
    
    # Yellow - stricter S-channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 120) & (s_channel <= 255)] = 1
    
    # Yellow - LAB B-channel (stricter range)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= 150) & (b_channel <= 190)] = 1
    
    # Yellow - RGB (must have high R AND G, low B)
    r_channel = img[:, :, 2]
    g_channel = img[:, :, 1]
    b_channel_rgb = img[:, :, 0]
    yellow_rgb = np.zeros_like(r_channel)
    yellow_rgb[(r_channel >= 200) & (g_channel >= 200) & (b_channel_rgb < 150)] = 1
    
    # Combine yellow detections (must pass 2 out of 3 tests)
    yellow_votes = s_binary.astype(int) + b_binary.astype(int) + yellow_rgb.astype(int)
    yellow = np.zeros_like(s_binary)
    yellow[yellow_votes >= 2] = 1
    
    # Final: yellow OR white
    combined = np.zeros_like(l_binary)
    combined[(yellow == 1) | (l_binary == 1)] = 1
    
    return combined

# Test
import sys
sys.path.insert(0, '.')
from lane_detection_final import load_calibration, combined_threshold as old_threshold, get_perspective_transform

def compare_thresholds(img_path):
    print(f"\n{img_path}")
    
    img = cv2.imread(img_path)
    mtx, dist = load_calibration()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # OLD
    binary_old = old_threshold(undist)
    
    # NEW
    binary_new = strict_threshold(undist)
    
    print(f"  OLD: {np.sum(binary_old):6d} pixels")
    print(f"  NEW: {np.sum(binary_new):6d} pixels ({np.sum(binary_new)/np.sum(binary_old)*100:.1f}%)")
    
    # Warp and check
    M, Minv = get_perspective_transform(img.shape)
    warped_old = cv2.warpPerspective(binary_old, M, (img.shape[1], img.shape[0]))
    warped_new = cv2.warpPerspective(binary_new, M, (img.shape[1], img.shape[0]))
    
    # Visualize
    h, w = img.shape[:2]
    comparison = np.hstack([
        cv2.resize(cv2.cvtColor(warped_old*255, cv2.COLOR_GRAY2BGR), (w//2, h//2)),
        cv2.resize(cv2.cvtColor(warped_new*255, cv2.COLOR_GRAY2BGR), (w//2, h//2))
    ])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "OLD (noisy)", (10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(comparison, "NEW (clean)", (w//2+10, 30), font, 0.7, (255,255,255), 2)
    
    import os
    basename = os.path.basename(img_path).replace('.jpg', '')
    cv2.imwrite(f'debug/strict_threshold_{basename}.jpg', comparison)
    print(f"  ✓ Saved: debug/strict_threshold_{basename}.jpg")

images = [
    'test_images/test1.jpg',
    'test_images/test4.jpg',
    'test_images/challange00101.jpg',
    'test_images/challange00111.jpg',
]

print("="*60)
print("STRICT THRESHOLD TEST")
print("="*60)

for img_path in images:
    compare_thresholds(img_path)

print("\n" + "="*60)
print("Strict threshold should have less noise, cleaner lines")
print("="*60)
