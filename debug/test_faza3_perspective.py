#!/usr/bin/env python3
"""
FAZA 3: Test Perspektivne Transformacije

Testira da li perspective transform pravilno transformiše sliku u bird's eye view.
"""

import numpy as np
import cv2
import os

def load_calibration():
    """Load camera calibration"""
    data = np.load('calibration.npz')
    return data['mtx'], data['dist']

def get_perspective_transform_points(img_shape):
    """
    Define source and destination points for perspective transform.
    
    Source: trapezoid on the road (perspective view)
    Destination: rectangle (bird's eye view)
    """
    h, w = img_shape[:2]
    
    # Source points - trapezoid around lane lines
    src = np.float32([
        [w * 0.45, h * 0.63],  # Top left
        [w * 0.55, h * 0.63],  # Top right
        [w * 0.15, h],         # Bottom left
        [w * 0.85, h]          # Bottom right
    ])
    
    # Destination points - rectangle
    dst = np.float32([
        [w * 0.25, 0],         # Top left
        [w * 0.75, 0],         # Top right
        [w * 0.25, h],         # Bottom left
        [w * 0.75, h]          # Bottom right
    ])
    
    return src, dst

def draw_roi(img, points, color=(255, 0, 0), thickness=3):
    """Draw ROI polygon on image"""
    img_copy = img.copy()
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_copy, [pts], True, color, thickness)
    
    # Draw points
    for i, pt in enumerate(points):
        cv2.circle(img_copy, tuple(pt.astype(int)), 10, color, -1)
        cv2.putText(img_copy, str(i+1), tuple((pt + 15).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return img_copy

def test_perspective_transform(img_path, mtx, dist):
    """Test perspective transform on a single image"""
    
    print(f"\nTesting: {os.path.basename(img_path)}")
    
    # Load and undistort
    img = cv2.imread(img_path)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Get transform points
    src, dst = get_perspective_transform_points(img.shape)
    
    print(f"  Source points (trapezoid):")
    for i, pt in enumerate(src):
        print(f"    {i+1}. ({pt[0]:.0f}, {pt[1]:.0f})")
    
    print(f"  Destination points (rectangle):")
    for i, pt in enumerate(dst):
        print(f"    {i+1}. ({pt[0]:.0f}, {pt[1]:.0f})")
    
    # Calculate transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp image
    warped = cv2.warpPerspective(undist, M, (img.shape[1], img.shape[0]), 
                                  flags=cv2.INTER_LINEAR)
    
    # Draw ROI on original
    img_with_roi = draw_roi(undist, src, color=(0, 255, 0))
    
    # Draw ROI on warped
    warped_with_roi = draw_roi(warped, dst, color=(0, 255, 0))
    
    # Create comparison
    h, w = img.shape[:2]
    
    # Top row: Original with ROI, Warped with ROI
    top_left = cv2.resize(img_with_roi, (w//2, h//2))
    top_right = cv2.resize(warped_with_roi, (w//2, h//2))
    top_row = np.hstack([top_left, top_right])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(top_row, "Original + ROI", (10, 40), font, 1, (255,255,255), 2)
    cv2.putText(top_row, "Warped (Bird's Eye)", (w//2+10, 40), font, 1, (255,255,255), 2)
    
    # Save
    basename = os.path.basename(img_path).replace('.jpg', '')
    output_path = f"debug/faza3_perspective_{basename}.jpg"
    cv2.imwrite(output_path, top_row)
    print(f"  ✓ Saved: {output_path}")
    
    return M, Minv

def test_faza3():
    """Test Phase 3: Perspective Transform"""
    
    print("=" * 60)
    print("FAZA 3: TEST PERSPEKTIVNE TRANSFORMACIJE")
    print("=" * 60)
    
    # Load calibration
    if not os.path.exists('calibration.npz'):
        print("✗ Nema calibration.npz - prvo pokreni FAZA 1")
        return False
    
    mtx, dist = load_calibration()
    print("✓ Učitana kalibracija")
    
    # Test on straight lines image - best for checking if transform is correct
    test_images = [
        'test_images/straight_lines1.jpg',
        'test_images/straight_lines2.jpg',
        'test_images/test1.jpg',
    ]
    
    print("\nTestiram perspektivnu transformaciju...\n")
    
    for img_path in test_images:
        if os.path.exists(img_path):
            M, Minv = test_perspective_transform(img_path, mtx, dist)
    
    # Save transform matrices
    np.savez('perspective_transform.npz', M=M, Minv=Minv)
    print("\n✓ Transform matrice sačuvane u perspective_transform.npz")
    
    print("\n" + "=" * 60)
    print("✓ FAZA 3 ZAVRŠENA")
    print("=" * 60)
    print("\n→ Proveri slike u debug/ folderu")
    print("→ Na warped slici, prave linije treba da budu PARALELNE i VERTIKALNE")
    print("→ Ako je OK, nastavi sa FAZA 4: Detekcija Piksela")
    
    return True

if __name__ == "__main__":
    os.makedirs('debug', exist_ok=True)
    test_faza3()
