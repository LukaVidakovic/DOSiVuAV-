#!/usr/bin/env python3
"""
FAZA 5: Test Fitovanja Polinoma

Testira fitovanje polinoma drugog reda kroz detektovane piksele.
"""

import numpy as np
import cv2
import os
import sys
sys.path.insert(0, '.')

# Import from previous phase
from debug.test_faza4_detection import (
    load_calibration, load_perspective_transform, 
    combined_threshold, find_lane_pixels_sliding_window
)

def fit_polynomial(leftx, lefty, rightx, righty):
    """
    Fit second order polynomial to lane pixels.
    
    Polynomial form: x = Ay² + By + C
    
    Returns:
        left_fit: [A, B, C] for left lane
        right_fit: [A, B, C] for right lane
    """
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit

def visualize_polynomial(binary_warped, leftx, lefty, rightx, righty, left_fit, right_fit):
    """Create visualization with fitted polynomial"""
    
    # Create output image
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # Color lane pixels
    out_img[lefty, leftx] = [255, 0, 0]  # Red
    out_img[righty, rightx] = [0, 0, 255]  # Blue
    
    # Generate y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    # Calculate x values from polynomial
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Draw polynomial lines
    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    
    cv2.polylines(out_img, np.int32([left_line_pts]), False, (255, 255, 0), thickness=5)
    cv2.polylines(out_img, np.int32([right_line_pts]), False, (255, 255, 0), thickness=5)
    
    # Fill lane area
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    overlay = out_img.copy()
    cv2.fillPoly(overlay, np.int32([pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 0.7, overlay, 0.3, 0)
    
    return out_img, left_fitx, right_fitx, ploty

def test_polynomial_fit(img_path, mtx, dist, M):
    """Test polynomial fitting on a single image"""
    
    print(f"\nTesting: {os.path.basename(img_path)}")
    
    # Load and undistort
    img = cv2.imread(img_path)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Apply thresholding and warp
    binary = combined_threshold(undist)
    binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
    
    # Find lane pixels
    leftx, lefty, rightx, righty, _ = find_lane_pixels_sliding_window(binary_warped)
    
    if len(leftx) < 100 or len(rightx) < 100:
        print(f"  ✗ Nedovoljno piksela za fitovanje")
        return False
    
    # Fit polynomial
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
    
    print(f"  Left polynomial:  x = {left_fit[0]:.2e}*y² + {left_fit[1]:.2e}*y + {left_fit[2]:.2f}")
    print(f"  Right polynomial: x = {right_fit[0]:.2e}*y² + {right_fit[1]:.2e}*y + {right_fit[2]:.2f}")
    
    # Visualize
    out_img, left_fitx, right_fitx, ploty = visualize_polynomial(
        binary_warped, leftx, lefty, rightx, righty, left_fit, right_fit
    )
    
    # Calculate lane width at bottom
    y_bottom = binary_warped.shape[0] - 1
    left_x_bottom = left_fit[0]*y_bottom**2 + left_fit[1]*y_bottom + left_fit[2]
    right_x_bottom = right_fit[0]*y_bottom**2 + right_fit[1]*y_bottom + right_fit[2]
    lane_width_pixels = right_x_bottom - left_x_bottom
    
    print(f"  Lane width at bottom: {lane_width_pixels:.0f} pixels")
    
    # Sanity check - lane width should be reasonable (400-900 pixels)
    if lane_width_pixels < 400 or lane_width_pixels > 900:
        print(f"  ⚠ Lane width izgleda nerazumno!")
    else:
        print(f"  ✓ Lane width izgleda OK")
    
    # Save visualization
    basename = os.path.basename(img_path).replace('.jpg', '')
    output_path = f"debug/faza5_polynomial_{basename}.jpg"
    cv2.imwrite(output_path, out_img)
    print(f"  ✓ Saved: {output_path}")
    
    return True

def test_faza5():
    """Test Phase 5: Polynomial Fitting"""
    
    print("=" * 60)
    print("FAZA 5: TEST FITOVANJA POLINOMA")
    print("=" * 60)
    
    # Load calibration and perspective transform
    if not os.path.exists('calibration.npz'):
        print("✗ Nema calibration.npz")
        return False
    
    if not os.path.exists('perspective_transform.npz'):
        print("✗ Nema perspective_transform.npz")
        return False
    
    mtx, dist = load_calibration()
    M, Minv = load_perspective_transform()
    print("✓ Učitana kalibracija i perspective transform")
    
    # Test images
    test_images = [
        'test_images/straight_lines1.jpg',
        'test_images/test1.jpg',
        'test_images/solidYellowLeft.jpg',
    ]
    
    print("\nTestiram fitovanje polinoma...\n")
    
    successes = 0
    total = 0
    
    for img_path in test_images:
        if os.path.exists(img_path):
            success = test_polynomial_fit(img_path, mtx, dist, M)
            total += 1
            if success:
                successes += 1
    
    print("\n" + "=" * 60)
    print(f"Uspešnost: {successes}/{total} slika")
    print("=" * 60)
    print("\n→ Proveri slike u debug/ folderu")
    print("→ Žute linije treba glatko da prate detektovane piksele")
    print("→ Zelena oblast treba da pokriva traku između linija")
    print("→ Ako je OK, nastavi sa FAZA 6: Zakrivljenost i Pozicija")
    
    return successes == total

if __name__ == "__main__":
    os.makedirs('debug', exist_ok=True)
    test_faza5()
