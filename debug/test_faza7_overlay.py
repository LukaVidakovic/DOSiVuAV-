#!/usr/bin/env python3
"""
FAZA 7: Test Vizualizacije i Overlay

Testira crtanje detektovane trake na originalnoj slici.
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
from debug.test_faza6_curvature import calculate_curvature, calculate_vehicle_position

def draw_lane_overlay(undist, binary_warped, left_fit, right_fit, Minv):
    """
    Draw detected lane area on original image.
    
    Args:
        undist: Undistorted original image
        binary_warped: Binary warped image (for shape)
        left_fit, right_fit: Polynomial coefficients
        Minv: Inverse perspective transform matrix
    
    Returns:
        result: Image with lane overlay
    """
    # Create image to draw lanes
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Generate y values
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    # Calculate x values from polynomials
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast points for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw lane onto warped blank image
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    
    # Draw lane lines
    cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), thickness=25)
    cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), thickness=25)
    
    # Warp back to original image space
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    
    # Combine with original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

def add_text_overlay(img, curvature, offset):
    """
    Add text overlay with metrics.
    
    Args:
        img: Input image
        curvature: Lane curvature radius (meters)
        offset: Vehicle offset from center (meters)
    
    Returns:
        Image with text overlay
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    color = (255, 255, 255)
    
    # Curvature text
    if curvature > 10000:
        curv_text = "Radius of Curvature: Straight"
    else:
        curv_text = f"Radius of Curvature: {curvature:.0f}m"
    
    cv2.putText(img, curv_text, (50, 50), font, font_scale, color, font_thickness)
    
    # Vehicle position text
    if abs(offset) < 0.05:
        pos_text = "Vehicle Position: Centered"
    elif offset < 0:
        pos_text = f"Vehicle Position: {abs(offset):.2f}m left of center"
    else:
        pos_text = f"Vehicle Position: {offset:.2f}m right of center"
    
    cv2.putText(img, pos_text, (50, 100), font, font_scale, color, font_thickness)
    
    return img

def test_visualization(img_path, mtx, dist, M, Minv):
    """Test complete visualization on a single image"""
    
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
    
    # Calculate metrics
    y_eval = binary_warped.shape[0] - 1
    left_curv, right_curv = calculate_curvature(left_fit, right_fit, y_eval, binary_warped.shape)
    avg_curv = (left_curv + right_curv) / 2
    offset = calculate_vehicle_position(left_fit, right_fit, img.shape[1])
    
    print(f"  Curvature: {avg_curv:.0f}m")
    print(f"  Offset: {offset:+.2f}m")
    
    # Draw lane overlay
    result = draw_lane_overlay(undist, binary_warped, left_fit, right_fit, Minv)
    
    # Add text
    result = add_text_overlay(result, avg_curv, offset)
    
    # Create comparison: original vs result
    comparison = np.hstack((undist, result))
    
    # Save
    basename = os.path.basename(img_path).replace('.jpg', '')
    output_path = f"debug/faza7_overlay_{basename}.jpg"
    cv2.imwrite(output_path, comparison)
    print(f"  ✓ Saved: {output_path}")
    
    # Also save just the result
    result_path = f"output_images/{basename}_final.jpg"
    cv2.imwrite(result_path, result)
    print(f"  ✓ Saved: {result_path}")
    
    return True

def test_faza7():
    """Test Phase 7: Visualization and Overlay"""
    
    print("=" * 60)
    print("FAZA 7: TEST VIZUALIZACIJE I OVERLAY")
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
    
    # Create output directory
    os.makedirs('output_images', exist_ok=True)
    
    # Test images
    test_images = [
        'test_images/straight_lines1.jpg',
        'test_images/test1.jpg',
        'test_images/test4.jpg',
        'test_images/solidYellowLeft.jpg',
    ]
    
    print("\nTestiram vizualizaciju...\n")
    
    successes = 0
    total = 0
    
    for img_path in test_images:
        if os.path.exists(img_path):
            success = test_visualization(img_path, mtx, dist, M, Minv)
            total += 1
            if success:
                successes += 1
    
    print("\n" + "=" * 60)
    print(f"Uspešnost: {successes}/{total} slika")
    print("=" * 60)
    print("\n→ Proveri slike u debug/ i output_images/ folderima")
    print("→ Zelena traka treba precizno da pokriva prostor između linija")
    print("→ Tekst treba da prikazuje razumne vrednosti")
    print("\n→ Ako je OK, nastavi sa FAZA 8: Pipeline za Slike")
    
    return successes == total

if __name__ == "__main__":
    os.makedirs('debug', exist_ok=True)
    test_faza7()
