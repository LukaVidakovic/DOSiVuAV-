#!/usr/bin/env python3
"""
FAZA 4: Test Detekcije Piksela Linija

Testira sliding window algoritam za pronalaženje piksela koji pripadaju linijama.
"""

import numpy as np
import cv2
import os

def load_calibration():
    """Load camera calibration"""
    data = np.load('calibration.npz')
    return data['mtx'], data['dist']

def load_perspective_transform():
    """Load perspective transform matrices"""
    data = np.load('perspective_transform.npz')
    return data['M'], data['Minv']

def combined_threshold(img):
    """Apply combined thresholding (from FAZA 2)"""
    # Sobel X
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
    
    # HLS L-channel (white)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 200) & (l_channel <= 255)] = 1
    
    # HLS S-channel (yellow)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 90) & (s_channel <= 255)] = 1
    
    # LAB B-channel (yellow)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= 155) & (b_channel <= 200)] = 1
    
    # Combine
    combined = np.zeros_like(sobel_binary)
    combined[((sobel_binary == 1) | (l_binary == 1)) & 
             ((sobel_binary == 1) | (s_binary == 1) | (b_binary == 1))] = 1
    
    return combined

def find_lane_pixels_sliding_window(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    Find lane pixels using sliding window search.
    
    Args:
        binary_warped: Binary warped image
        nwindows: Number of sliding windows
        margin: Width of windows (+/- margin)
        minpix: Minimum pixels to recenter window
    
    Returns:
        leftx, lefty, rightx, righty: Pixel coordinates
        out_img: Visualization image
    """
    # Take histogram of bottom half
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    
    # Create output image
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # Find peaks in left and right halves
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    print(f"  Histogram peaks: left={leftx_base}, right={rightx_base}, width={rightx_base-leftx_base}")
    
    # Set height of windows
    window_height = binary_warped.shape[0] // nwindows
    
    # Identify x and y positions of all nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Lists to store lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through windows
    for window in range(nwindows):
        # Identify window boundaries
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw windows on visualization
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
                     (win_xleft_high, win_y_high), (0, 255, 0), 3)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), 
                     (win_xright_high, win_y_high), (0, 255, 0), 3)
        
        # Identify nonzero pixels in window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append indices
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Recenter if enough pixels found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Color lane pixels
    out_img[lefty, leftx] = [255, 0, 0]  # Red for left
    out_img[righty, rightx] = [0, 0, 255]  # Blue for right
    
    return leftx, lefty, rightx, righty, out_img

def test_lane_detection(img_path, mtx, dist, M):
    """Test lane detection on a single image"""
    
    print(f"\nTesting: {os.path.basename(img_path)}")
    
    # Load and undistort
    img = cv2.imread(img_path)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Apply thresholding
    binary = combined_threshold(undist)
    print(f"  Binary pixels: {np.sum(binary)}")
    
    # Warp to bird's eye view
    binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
    print(f"  Warped binary pixels: {np.sum(binary_warped)}")
    
    # Find lane pixels
    leftx, lefty, rightx, righty, out_img = find_lane_pixels_sliding_window(binary_warped)
    
    print(f"  Left lane pixels: {len(leftx)}")
    print(f"  Right lane pixels: {len(rightx)}")
    
    # Check if detection is successful
    if len(leftx) < 100 or len(rightx) < 100:
        print(f"  ✗ Nedovoljno piksela detektovano!")
        success = False
    else:
        print(f"  ✓ Detekcija uspešna")
        success = True
    
    # Save visualization
    basename = os.path.basename(img_path).replace('.jpg', '')
    output_path = f"debug/faza4_detection_{basename}.jpg"
    cv2.imwrite(output_path, out_img)
    print(f"  ✓ Saved: {output_path}")
    
    return success, leftx, lefty, rightx, righty

def test_faza4():
    """Test Phase 4: Lane Pixel Detection"""
    
    print("=" * 60)
    print("FAZA 4: TEST DETEKCIJE PIKSELA LINIJA")
    print("=" * 60)
    
    # Load calibration and perspective transform
    if not os.path.exists('calibration.npz'):
        print("✗ Nema calibration.npz - prvo pokreni FAZA 1")
        return False
    
    if not os.path.exists('perspective_transform.npz'):
        print("✗ Nema perspective_transform.npz - prvo pokreni FAZA 3")
        return False
    
    mtx, dist = load_calibration()
    M, Minv = load_perspective_transform()
    print("✓ Učitana kalibracija i perspective transform")
    
    # Test images
    test_images = [
        'test_images/straight_lines1.jpg',
        'test_images/test1.jpg',
        'test_images/solidWhiteRight.jpg',
        'test_images/solidYellowLeft.jpg',
    ]
    
    print("\nTestiram sliding window detekciju...\n")
    
    successes = 0
    total = 0
    
    for img_path in test_images:
        if os.path.exists(img_path):
            success, _, _, _, _ = test_lane_detection(img_path, mtx, dist, M)
            total += 1
            if success:
                successes += 1
    
    print("\n" + "=" * 60)
    print(f"Uspešnost: {successes}/{total} slika")
    print("=" * 60)
    print("\n→ Proveri slike u debug/ folderu")
    print("→ Zeleni prozori treba da prate linije")
    print("→ Crveni (leva) i plavi (desna) pikseli treba da pokrivaju linije")
    print("→ Ako je OK, nastavi sa FAZA 5: Fitovanje Polinoma")
    
    return successes == total

if __name__ == "__main__":
    os.makedirs('debug', exist_ok=True)
    test_faza4()
