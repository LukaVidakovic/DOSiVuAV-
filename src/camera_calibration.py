"""
Camera Calibration Module

This module provides functions to calibrate a camera using chessboard images
and apply distortion correction to images.

Based on the lecture example: primeri_sa_predavanja/09_01_calibration.py
"""

import numpy as np
import cv2
import glob
import os
from typing import Tuple


def calibrate_camera(
    images_path: str = "camera_cal/calibration*.jpg",
    chessboard_size: Tuple[int, int] = (9, 6),
    save_path: str = "calibration.npz"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrate camera using chessboard images.

    Args:
        images_path: Glob pattern for calibration images
        chessboard_size: Number of inner corners (cols, rows)
        save_path: Path to save calibration data

    Returns:
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Define the chess board cols and rows
    cols, rows = chessboard_size

    # Set the termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points: (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Get list of calibration images
    images = glob.glob(images_path)

    if not images:
        raise FileNotFoundError(f"No calibration images found at: {images_path}")

    print(f"Found {len(images)} calibration images")

    # Image size (will be set from first successful image)
    img_size = None

    # Loop over calibration images
    successful_calibrations = 0
    for fname in images:
        img = cv2.imread(fname)

        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Set image size from first image
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        # If found, add object points and image points
        if ret:
            objpoints.append(objp)

            # Refine corner positions to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(corners_refined)

            successful_calibrations += 1
            print(f"✓ Calibration {successful_calibrations}: {os.path.basename(fname)}")
        else:
            print(f"✗ Failed to find corners: {os.path.basename(fname)}")

    if successful_calibrations == 0:
        raise RuntimeError("No successful calibrations found")

    print(f"\nSuccessfully calibrated using {successful_calibrations}/{len(images)} images")

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # Calculate calibration error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"Mean re-projection error: {mean_error:.4f} pixels")

    # Save calibration data
    if save_path:
        save_calibration(mtx, dist, rvecs, tvecs, save_path)
        print(f"Calibration saved to: {save_path}")

    return mtx, dist, rvecs, tvecs


def save_calibration(
    mtx: np.ndarray,
    dist: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    save_path: str = "calibration.npz"
) -> None:
    """
    Save camera calibration parameters to file.

    Args:
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
        save_path: Path to save calibration file
    """
    np.savez(save_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


def load_calibration(
    load_path: str = "calibration.npz"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load camera calibration parameters from file.

    Args:
        load_path: Path to calibration file

    Returns:
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Calibration file not found: {load_path}")

    calibration = np.load(load_path)
    mtx = calibration['mtx']
    dist = calibration['dist']
    rvecs = calibration['rvecs']
    tvecs = calibration['tvecs']

    return mtx, dist, rvecs, tvecs


def undistort_image(
    img: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
    optimize: bool = True
) -> np.ndarray:
    """
    Apply distortion correction to an image.

    Args:
        img: Input image (distorted)
        mtx: Camera matrix
        dist: Distortion coefficients
        optimize: Whether to use optimal new camera matrix

    Returns:
        Undistorted image
    """
    h, w = img.shape[:2]

    if optimize:
        # Get optimal new camera matrix
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h)
        )

        # Undistort with optimal matrix
        undist = cv2.undistort(img, mtx, dist, None, newcameramtx)
    else:
        # Undistort without optimization
        undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist


if __name__ == "__main__":
    """
    Example usage: Run camera calibration
    """
    print("=" * 60)
    print("Camera Calibration")
    print("=" * 60)

    # Calibrate camera using chessboard images
    # Note: OpenCV finds chessboard by inner corners
    # Our chessboard is 9x6 inner corners (10x7 squares)
    mtx, dist, rvecs, tvecs = calibrate_camera(
        images_path="camera_cal/calibration*.jpg",
        chessboard_size=(9, 6),
        save_path="calibration.npz"
    )

    print("\n" + "=" * 60)
    print("Camera Matrix:")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)
    print("=" * 60)

    # Test undistortion on a test image
    print("\nTesting undistortion on test image...")
    test_img = cv2.imread("test_images/test1.jpg")

    if test_img is not None:
        undistorted = undistort_image(test_img, mtx, dist)

        # Save comparison
        comparison = np.hstack((test_img, undistorted))
        cv2.imwrite("docs/undistort_comparison.jpg", comparison)
        print("✓ Saved undistortion comparison to: docs/undistort_comparison.jpg")
    else:
        print("✗ Could not load test image")

    print("\n✓ Camera calibration complete!")
