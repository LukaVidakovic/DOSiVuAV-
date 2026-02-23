#!/usr/bin/env python3
"""Poboljšan threshold - bolja detekcija žute, manje šuma od ograde"""

import numpy as np
import cv2

def improved_threshold(img):
    """Improved thresholding focusing on yellow/white lanes, rejecting gray barriers"""
    
    # 1. Sobel X - edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= 30) & (scaled_sobel <= 100)] = 1  # Povećan min threshold
    
    # 2. HLS L-channel - white lines
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 220) & (l_channel <= 255)] = 1  # Strožije za belu
    
    # 3. HLS S-channel - yellow lines (KLJUČNO!)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 100) & (s_channel <= 255)] = 1  # Povećan min
    
    # 4. LAB B-channel - yellow (KLJUČNO!)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= 145) & (b_channel <= 200)] = 1  # Prošireno
    
    # 5. RGB - dodatna provera za žutu
    r_channel = img[:, :, 2]
    g_channel = img[:, :, 1]
    yellow_rgb = np.zeros_like(r_channel)
    yellow_rgb[(r_channel >= 180) & (g_channel >= 180)] = 1  # Žuta je R+G
    
    # Kombinacija: fokus na žutu i belu, ne na sivu ogradu
    # Žuta: S ILI B ILI yellow_rgb
    yellow = np.zeros_like(s_binary)
    yellow[(s_binary == 1) | (b_binary == 1) | (yellow_rgb == 1)] = 1
    
    # Bela: L
    white = l_binary
    
    # Finalna kombinacija
    combined = np.zeros_like(sobel_binary)
    combined[(yellow == 1) | (white == 1)] = 1
    
    return combined

# Test
import sys
sys.path.insert(0, '.')
from debug.fix_roi import load_calibration, get_perspective_transform_NEW

def test_improved(img_path):
    print(f"\nTesting: {img_path}")
    
    img = cv2.imread(img_path)
    mtx, dist = load_calibration()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Old threshold
    from debug.fix_roi import combined_threshold as old_threshold
    binary_old = old_threshold(undist)
    
    # New threshold
    binary_new = improved_threshold(undist)
    
    print(f"  OLD threshold: {np.sum(binary_old):6d} pixels")
    print(f"  NEW threshold: {np.sum(binary_new):6d} pixels")
    
    # Warp
    M, Minv, src, dst = get_perspective_transform_NEW(img.shape)
    warped_old = cv2.warpPerspective(binary_old, M, (img.shape[1], img.shape[0]))
    warped_new = cv2.warpPerspective(binary_new, M, (img.shape[1], img.shape[0]))
    
    # Histogram
    hist_old = np.sum(warped_old[warped_old.shape[0]//2:, :], axis=0)
    hist_new = np.sum(warped_new[warped_new.shape[0]//2:, :], axis=0)
    
    midpoint = len(hist_old) // 2
    left_old = np.argmax(hist_old[:midpoint])
    right_old = np.argmax(hist_old[midpoint:]) + midpoint
    left_new = np.argmax(hist_new[:midpoint])
    right_new = np.argmax(hist_new[midpoint:]) + midpoint
    
    print(f"  OLD hist: left={left_old:3d}, right={right_old:4d}")
    print(f"  NEW hist: left={left_new:3d}, right={right_new:4d}")
    
    # Visualize
    h, w = img.shape[:2]
    
    warped_old_vis = cv2.cvtColor(warped_old*255, cv2.COLOR_GRAY2BGR)
    cv2.line(warped_old_vis, (left_old, 0), (left_old, h), (0, 255, 0), 2)
    cv2.line(warped_old_vis, (right_old, 0), (right_old, h), (0, 0, 255), 2)
    
    warped_new_vis = cv2.cvtColor(warped_new*255, cv2.COLOR_GRAY2BGR)
    cv2.line(warped_new_vis, (left_new, 0), (left_new, h), (0, 255, 0), 2)
    cv2.line(warped_new_vis, (right_new, 0), (right_new, h), (0, 0, 255), 2)
    
    comparison = np.hstack([cv2.resize(warped_old_vis, (w//2, h//2)), 
                           cv2.resize(warped_new_vis, (w//2, h//2))])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "OLD Threshold", (10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(comparison, "NEW Threshold", (w//2+10, 30), font, 0.7, (255,255,255), 2)
    
    import os
    basename = os.path.basename(img_path).replace('.jpg', '')
    cv2.imwrite(f'debug/threshold_comparison_{basename}.jpg', comparison)
    print(f"  ✓ Saved: debug/threshold_comparison_{basename}.jpg")

images = [
    'test_images/test1.jpg',
    'test_images/test4.jpg',
    'test_images/challange00101.jpg',
]

print("="*60)
print("TESTIRANJE POBOLJŠANOG THRESHOLD")
print("="*60)

for img_path in images:
    test_improved(img_path)

print("\n" + "="*60)
print("Proveri debug/ folder - NEW threshold treba bolje da detektuje žutu liniju")
print("="*60)
