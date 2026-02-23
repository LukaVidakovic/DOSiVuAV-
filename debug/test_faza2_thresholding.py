#!/usr/bin/env python3
"""
FAZA 2: Test Binarne Segmentacije

Testira različite metode thresholding-a da pronađemo najbolju kombinaciju.
"""

import numpy as np
import cv2
import os

def load_calibration():
    """Load camera calibration"""
    data = np.load('calibration.npz')
    return data['mtx'], data['dist']

def sobel_x_threshold(img, thresh=(20, 100)):
    """Sobel X gradient threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary

def hls_s_threshold(img, thresh=(90, 255)):
    """HLS S-channel threshold (yellow lines)"""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    
    binary = np.zeros_like(s_channel)
    binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary

def hls_l_threshold(img, thresh=(200, 255)):
    """HLS L-channel threshold (white lines)"""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    
    binary = np.zeros_like(l_channel)
    binary[(l_channel >= thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary

def lab_b_threshold(img, thresh=(155, 200)):
    """LAB B-channel threshold (yellow lines)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    
    binary = np.zeros_like(b_channel)
    binary[(b_channel >= thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary

def test_single_image(img_path, mtx, dist):
    """Test thresholding on a single image"""
    
    print(f"\nTesting: {os.path.basename(img_path)}")
    
    # Load and undistort
    img = cv2.imread(img_path)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Test different thresholds
    sobel = sobel_x_threshold(undist)
    hls_s = hls_s_threshold(undist)
    hls_l = hls_l_threshold(undist)
    lab_b = lab_b_threshold(undist)
    
    # Count pixels
    print(f"  Sobel X:     {np.sum(sobel):6d} pixels")
    print(f"  HLS S:       {np.sum(hls_s):6d} pixels")
    print(f"  HLS L:       {np.sum(hls_l):6d} pixels")
    print(f"  LAB B:       {np.sum(lab_b):6d} pixels")
    
    # Try different combinations
    combo1 = np.zeros_like(sobel)
    combo1[(sobel == 1) | (hls_s == 1)] = 1
    
    combo2 = np.zeros_like(sobel)
    combo2[(sobel == 1) | (hls_l == 1)] = 1
    
    combo3 = np.zeros_like(sobel)
    combo3[(hls_s == 1) | (hls_l == 1)] = 1
    
    combo4 = np.zeros_like(sobel)
    combo4[((sobel == 1) | (hls_l == 1)) & ((sobel == 1) | (hls_s == 1) | (lab_b == 1))] = 1
    
    print(f"  Combo1 (Sobel|S):        {np.sum(combo1):6d} pixels")
    print(f"  Combo2 (Sobel|L):        {np.sum(combo2):6d} pixels")
    print(f"  Combo3 (S|L):            {np.sum(combo3):6d} pixels")
    print(f"  Combo4 (Sobel|L)&(S|B):  {np.sum(combo4):6d} pixels")
    
    # Create visualization
    h, w = img.shape[:2]
    
    # Row 1: Original, Sobel, HLS-S
    row1_1 = cv2.resize(undist, (w//3, h//3))
    row1_2 = cv2.resize(cv2.cvtColor(sobel*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row1_3 = cv2.resize(cv2.cvtColor(hls_s*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row1 = np.hstack([row1_1, row1_2, row1_3])
    
    # Row 2: HLS-L, LAB-B, Combo1
    row2_1 = cv2.resize(cv2.cvtColor(hls_l*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row2_2 = cv2.resize(cv2.cvtColor(lab_b*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row2_3 = cv2.resize(cv2.cvtColor(combo1*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row2 = np.hstack([row2_1, row2_2, row2_3])
    
    # Row 3: Combo2, Combo3, Combo4
    row3_1 = cv2.resize(cv2.cvtColor(combo2*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row3_2 = cv2.resize(cv2.cvtColor(combo3*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row3_3 = cv2.resize(cv2.cvtColor(combo4*255, cv2.COLOR_GRAY2BGR), (w//3, h//3))
    row3 = np.hstack([row3_1, row3_2, row3_3])
    
    grid = np.vstack([row1, row2, row3])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid, "Original", (10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "Sobel X", (w//3+10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "HLS S", (2*w//3+10, 30), font, 0.7, (255,255,255), 2)
    
    cv2.putText(grid, "HLS L", (10, h//3+30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "LAB B", (w//3+10, h//3+30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "Sobel|S", (2*w//3+10, h//3+30), font, 0.7, (255,255,255), 2)
    
    cv2.putText(grid, "Sobel|L", (10, 2*h//3+30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "S|L", (w//3+10, 2*h//3+30), font, 0.7, (255,255,255), 2)
    cv2.putText(grid, "(Sobel|L)&(S|B)", (2*w//3+10, 2*h//3+30), font, 0.6, (255,255,255), 2)
    
    # Save
    basename = os.path.basename(img_path).replace('.jpg', '')
    output_path = f"debug/faza2_threshold_{basename}.jpg"
    cv2.imwrite(output_path, grid)
    print(f"  ✓ Saved: {output_path}")
    
    return combo4  # Return best combination

def test_faza2():
    """Test Phase 2: Binary Thresholding"""
    
    print("=" * 60)
    print("FAZA 2: TEST BINARNE SEGMENTACIJE")
    print("=" * 60)
    
    # Load calibration
    if not os.path.exists('calibration.npz'):
        print("✗ Nema calibration.npz - prvo pokreni FAZA 1")
        return False
    
    mtx, dist = load_calibration()
    print("✓ Učitana kalibracija")
    
    # Test images - različiti uslovi
    test_images = [
        'test_images/straight_lines1.jpg',  # Prave linije
        'test_images/test1.jpg',            # Zakrivljene, senka
        'test_images/solidWhiteRight.jpg',  # Bela linija
        'test_images/solidYellowLeft.jpg',  # Žuta linija
    ]
    
    print("\nTestiram različite threshold metode...\n")
    
    for img_path in test_images:
        if os.path.exists(img_path):
            test_single_image(img_path, mtx, dist)
    
    print("\n" + "=" * 60)
    print("✓ FAZA 2 ZAVRŠENA")
    print("=" * 60)
    print("\n→ Proveri slike u debug/ folderu")
    print("→ Linije treba da budu jasno vidljive, minimalan šum")
    print("→ Ako je OK, nastavi sa FAZA 3: Perspektivna Transformacija")
    
    return True

if __name__ == "__main__":
    os.makedirs('debug', exist_ok=True)
    test_faza2()
