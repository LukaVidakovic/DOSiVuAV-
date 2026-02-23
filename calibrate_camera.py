#!/usr/bin/env python3
"""
Camera Calibration - Step 1

Uses chessboard images from camera_cal/ folder to calculate
camera matrix and distortion coefficients.

Output: calibration.npz file
"""

import numpy as np
import cv2
import glob
import os

def calibrate_camera(images_path='camera_cal/calibration*.jpg', nx=9, ny=6):
    """
    Camera calibration using chessboard images
    
    Args:
        images_path: glob pattern for chessboard images
        nx: number of corners in width
        ny: number of corners in height
    
    Returns:
        mtx: camera matrix
        dist: distortion coefficients
        successful_images: number of successfully processed images
    """
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Load all chessboard images
    images = glob.glob(images_path)
    
    print(f"Found {len(images)} images for calibration...")
    
    successful = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            successful += 1
            print(f"  ✓ {os.path.basename(fname)}")
        else:
            print(f"  ✗ {os.path.basename(fname)} - corners not found")
    
    print(f"\nSuccessfully processed: {successful}/{len(images)} images")
    
    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    return mtx, dist, successful

def save_calibration(mtx, dist, filename='calibration.npz'):
    """Save calibration parameters"""
    np.savez(filename, mtx=mtx, dist=dist)
    print(f"\n✓ Calibration saved to {filename}")

def test_calibration(mtx, dist, test_image='camera_cal/calibration1.jpg'):
    """Test calibration on one image"""
    img = cv2.imread(test_image)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Save result
    output_path = 'docs/calibration_example.jpg'
    os.makedirs('docs', exist_ok=True)
    
    # Create side-by-side comparison
    combined = np.hstack((img, undist))
    cv2.imwrite(output_path, combined)
    print(f"✓ Calibration test saved to {output_path}")

if __name__ == '__main__':
    print("="*60)
    print("CAMERA CALIBRATION")
    print("="*60)
    
    # Calibrate
    mtx, dist, successful = calibrate_camera()
    
    if successful < 10:
        print("\n⚠ Warning: Few successful images for calibration!")
    
    # Save
    save_calibration(mtx, dist)
    
    # Test
    test_calibration(mtx, dist)
    
    print("\n" + "="*60)
    print("Calibration complete!")
    print("="*60)
