#!/usr/bin/env python3
"""
Popravljeni perspective transform koji radi sa različitim rezolucijama
"""

import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

from debug.test_faza4_detection import load_calibration, combined_threshold

def get_adaptive_perspective_transform(img_shape):
    """
    Get perspective transform that adapts to image resolution.
    
    Uses relative coordinates (percentages) instead of absolute pixels.
    """
    h, w = img_shape[:2]
    
    # Source points - trapezoid (adjusted for better coverage)
    src = np.float32([
        [w * 0.42, h * 0.65],  # Top left (wider)
        [w * 0.58, h * 0.65],  # Top right (wider)
        [w * 0.10, h * 0.95],  # Bottom left (not all the way down)
        [w * 0.90, h * 0.95]   # Bottom right
    ])
    
    # Destination points - rectangle
    dst = np.float32([
        [w * 0.20, 0],         # Top left
        [w * 0.80, 0],         # Top right
        [w * 0.20, h],         # Bottom left
        [w * 0.80, h]          # Bottom right
    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return M, Minv, src, dst

def test_on_image(img_path):
    """Test improved perspective transform"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {img_path}")
    print('='*60)
    
    # Load
    img = cv2.imread(img_path)
    print(f"Image shape: {img.shape}")
    
    # Undistort
    mtx, dist = load_calibration()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Threshold
    binary = combined_threshold(undist)
    print(f"Binary pixels: {np.sum(binary)}")
    
    # Get adaptive transform
    M, Minv, src, dst = get_adaptive_perspective_transform(img.shape)
    
    print(f"\nSource points:")
    for i, pt in enumerate(src):
        print(f"  {i+1}. ({pt[0]:.0f}, {pt[1]:.0f})")
    
    # Warp
    binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
    print(f"\nWarped binary pixels: {np.sum(binary_warped)}")
    
    # Histogram
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    print(f"Histogram peaks: left={leftx_base}, right={rightx_base}, width={rightx_base-leftx_base}")
    
    # Visualize
    h, w = img.shape[:2]
    
    # Draw ROI on original
    img_with_roi = undist.copy()
    pts = src.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_with_roi, [pts], True, (0, 255, 0), 3)
    
    # Warped with histogram peaks
    warped_vis = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
    cv2.line(warped_vis, (leftx_base, 0), (leftx_base, h), (0, 255, 0), 3)
    cv2.line(warped_vis, (rightx_base, 0), (rightx_base, h), (0, 0, 255), 3)
    
    # Create comparison
    row1 = np.hstack([cv2.resize(img_with_roi, (w//2, h//2)), 
                      cv2.resize(warped_vis, (w//2, h//2))])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(row1, "Original + ROI", (10, 30), font, 0.7, (255,255,255), 2)
    cv2.putText(row1, "Warped + Peaks", (w//2+10, 30), font, 0.7, (255,255,255), 2)
    
    # Save
    import os
    basename = os.path.basename(img_path).replace('.jpg', '')
    output_path = f"debug/improved_perspective_{basename}.jpg"
    cv2.imwrite(output_path, row1)
    print(f"\n✓ Saved: {output_path}")
    
    # Check if improved
    if leftx_base > 50 and rightx_base < w - 50:
        print("✓ Linije su dobro pozicionirane!")
        return True
    else:
        print("✗ Još uvek ima problema")
        return False

# Test on problematic images
test_images = [
    'test_images/solidYellowLeft.jpg',
    'test_images/solidWhiteRight.jpg',
    'test_images/straight_lines1.jpg',
    'test_images/test1.jpg',
]

print("\n" + "="*60)
print("TESTIRANJE POBOLJŠANOG PERSPECTIVE TRANSFORM")
print("="*60)

successes = 0
for img_path in test_images:
    if test_on_image(img_path):
        successes += 1

print(f"\n{'='*60}")
print(f"Uspešnost: {successes}/{len(test_images)}")
print("="*60)

# Save improved transform
M, Minv, src, dst = get_adaptive_perspective_transform((720, 1280, 3))
np.savez('perspective_transform_improved.npz', M=M, Minv=Minv)
print("\n✓ Improved transform saved to perspective_transform_improved.npz")
print("  (Note: This is for 720p reference, use get_adaptive_perspective_transform() for other resolutions)")
