#!/usr/bin/env python3
"""
FAZA 6: Test Računanja Zakrivljenosti i Pozicije Vozila

Testira računanje radijusa zakrivljenosti i pozicije vozila u metrima.
"""

import numpy as np
import cv2
import os
import sys
sys.path.insert(0, '.')

from debug.test_faza4_detection import (
    load_calibration, load_perspective_transform, 
    combined_threshold, find_lane_pixels_sliding_window
)
from debug.test_faza5_polynomial import fit_polynomial

# Conversion factors: pixels to meters
YM_PER_PIX = 30/720  # meters per pixel in y dimension (30m lane length / 720 pixels)
XM_PER_PIX = 3.7/700  # meters per pixel in x dimension (3.7m lane width / 700 pixels)

def calculate_curvature(left_fit, right_fit, y_eval, img_shape):
    """
    Calculate radius of curvature in meters.
    
    Formula: R = (1 + (2Ay + B)²)^(3/2) / |2A|
    
    Args:
        left_fit, right_fit: Polynomial coefficients in pixel space
        y_eval: Y position where to evaluate (typically bottom of image)
        img_shape: Image shape for generating points
    
    Returns:
        left_curverad, right_curverad: Curvature radius in meters
    """
    # Generate y values
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    # Calculate x values from polynomials
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Fit polynomial in world space (meters)
    left_fit_cr = np.polyfit(ploty*YM_PER_PIX, left_fitx*XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(ploty*YM_PER_PIX, right_fitx*XM_PER_PIX, 2)
    
    # Calculate curvature at y_eval
    y_eval_world = y_eval * YM_PER_PIX
    
    # R = (1 + (2Ay + B)²)^(3/2) / |2A|
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_world + left_fit_cr[1])**2)**1.5) / \
                    np.abs(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_world + right_fit_cr[1])**2)**1.5) / \
                     np.abs(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def calculate_vehicle_position(left_fit, right_fit, img_width):
    """
    Calculate vehicle position relative to lane center.
    
    Assumes camera is mounted at center of vehicle.
    
    Args:
        left_fit, right_fit: Polynomial coefficients
        img_width: Image width in pixels
    
    Returns:
        offset: Vehicle offset from center in meters
                Negative = left of center, Positive = right of center
    """
    # Calculate lane positions at bottom of image
    y_max = 720  # Assume 720p image
    
    left_lane_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    right_lane_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
    
    # Calculate lane center
    lane_center = (left_lane_pos + right_lane_pos) / 2
    
    # Calculate vehicle center (camera is centered)
    vehicle_center = img_width / 2
    
    # Calculate offset in meters
    offset = (vehicle_center - lane_center) * XM_PER_PIX
    
    return offset

def test_curvature_and_position(img_path, mtx, dist, M):
    """Test curvature and position calculation on a single image"""
    
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
        print(f"  ✗ Nedovoljno piksela")
        return False
    
    # Fit polynomial
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
    
    # Calculate curvature
    y_eval = binary_warped.shape[0] - 1  # Bottom of image
    left_curv, right_curv = calculate_curvature(left_fit, right_fit, y_eval, binary_warped.shape)
    avg_curv = (left_curv + right_curv) / 2
    
    print(f"  Left curvature:  {left_curv:7.0f} m")
    print(f"  Right curvature: {right_curv:7.0f} m")
    print(f"  Average:         {avg_curv:7.0f} m")
    
    # Sanity check for curvature
    if avg_curv < 50:
        print(f"  ⚠ Zakrivljenost previše mala (oštra krivina)")
    elif avg_curv > 10000:
        print(f"  ✓ Skoro prava linija")
    else:
        print(f"  ✓ Razumna zakrivljenost")
    
    # Calculate vehicle position
    offset = calculate_vehicle_position(left_fit, right_fit, img.shape[1])
    
    print(f"  Vehicle offset:  {offset:+.2f} m", end="")
    if abs(offset) < 0.1:
        print(" (centrirano)")
    elif offset < 0:
        print(f" ({abs(offset):.2f}m levo od centra)")
    else:
        print(f" ({offset:.2f}m desno od centra)")
    
    # Sanity check for offset
    if abs(offset) > 2.0:
        print(f"  ⚠ Offset izgleda prevelik!")
    else:
        print(f"  ✓ Offset izgleda OK")
    
    return True

def test_faza6():
    """Test Phase 6: Curvature and Vehicle Position"""
    
    print("=" * 60)
    print("FAZA 6: TEST ZAKRIVLJENOSTI I POZICIJE VOZILA")
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
    
    print(f"\nKonverzioni faktori:")
    print(f"  Y: {YM_PER_PIX:.6f} m/pixel (30m / 720px)")
    print(f"  X: {XM_PER_PIX:.6f} m/pixel (3.7m / 700px)")
    
    # Test images
    test_images = [
        'test_images/straight_lines1.jpg',  # Should have large curvature (straight)
        'test_images/test1.jpg',            # Should have smaller curvature (curved)
        'test_images/solidYellowLeft.jpg',
    ]
    
    print("\nTestiram računanje zakrivljenosti i pozicije...\n")
    
    successes = 0
    total = 0
    
    for img_path in test_images:
        if os.path.exists(img_path):
            success = test_curvature_and_position(img_path, mtx, dist, M)
            total += 1
            if success:
                successes += 1
    
    print("\n" + "=" * 60)
    print(f"Uspešnost: {successes}/{total} slika")
    print("=" * 60)
    print("\n→ Proveri da li vrednosti imaju smisla:")
    print("  - Prave linije: zakrivljenost > 1000m")
    print("  - Zakrivljene linije: zakrivljenost 100-1000m")
    print("  - Pozicija vozila: -1m do +1m od centra")
    print("\n→ Ako je OK, nastavi sa FAZA 7: Vizualizacija i Overlay")
    
    return successes == total

if __name__ == "__main__":
    os.makedirs('debug', exist_ok=True)
    test_faza6()
