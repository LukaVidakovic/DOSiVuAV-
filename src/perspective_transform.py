"""
Perspective Transform Module

This module provides functions to transform images to bird's-eye view
for easier lane detection and measurement.

Based on lecture example: examples/example.py
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def get_default_src_points(img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Get default source points for perspective transform (trapezoid on road).

    Args:
        img_shape: Image shape (height, width)

    Returns:
        Source points as float32 array of shape (4, 2)
    """
    h, w = img_shape[:2]

    # Define trapezoid in source image
    # These points define a region that should be transformed to a rectangle
    # Adjust these based on camera mounting and road perspective
    src = np.float32([
        [w * 0.45, h * 0.63],  # Top-left
        [w * 0.55, h * 0.63],  # Top-right
        [w * 0.85, h * 0.95],  # Bottom-right
        [w * 0.15, h * 0.95]   # Bottom-left
    ])

    return src


def get_default_dst_points(img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Get default destination points for perspective transform (rectangle).

    Args:
        img_shape: Image shape (height, width)

    Returns:
        Destination points as float32 array of shape (4, 2)
    """
    h, w = img_shape[:2]

    # Define rectangle in destination image (bird's-eye view)
    dst = np.float32([
        [w * 0.25, 0],       # Top-left
        [w * 0.75, 0],       # Top-right
        [w * 0.75, h],       # Bottom-right
        [w * 0.25, h]        # Bottom-left
    ])

    return dst


def get_perspective_transform(
    src: np.ndarray,
    dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute perspective transform matrices.

    Args:
        src: Source points (4, 2) array
        dst: Destination points (4, 2) array

    Returns:
        M: Forward transform matrix (warp)
        Minv: Inverse transform matrix (unwarp)
    """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


def warp_image(
    img: np.ndarray,
    M: np.ndarray,
    img_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Apply perspective transform to warp image to bird's-eye view.

    Args:
        img: Input image
        M: Transform matrix
        img_size: Output image size (width, height). If None, uses input size.

    Returns:
        Warped image
    """
    if img_size is None:
        img_size = (img.shape[1], img.shape[0])

    warped = cv2.warpPerspective(
        img,
        M,
        img_size,
        flags=cv2.INTER_LINEAR
    )

    return warped


def unwarp_image(
    img: np.ndarray,
    Minv: np.ndarray,
    img_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Apply inverse perspective transform to unwarp image back to original view.

    Args:
        img: Warped image
        Minv: Inverse transform matrix
        img_size: Output image size (width, height). If None, uses input size.

    Returns:
        Unwarped image
    """
    if img_size is None:
        img_size = (img.shape[1], img.shape[0])

    unwarped = cv2.warpPerspective(
        img,
        Minv,
        img_size,
        flags=cv2.INTER_LINEAR
    )

    return unwarped


def draw_transform_region(
    img: np.ndarray,
    src: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw the transform region on the image for visualization.

    Args:
        img: Input image
        src: Source points (4, 2) array
        color: Line color (B, G, R)
        thickness: Line thickness

    Returns:
        Image with region drawn
    """
    img_copy = img.copy()

    # Convert points to integer
    pts = src.astype(np.int32)

    # Draw polygon
    cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness)

    return img_copy


def visualize_transform(
    img: np.ndarray,
    src: Optional[np.ndarray] = None,
    dst: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualize the perspective transform by showing source region and warped result.

    Args:
        img: Input image
        src: Source points. If None, uses default.
        dst: Destination points. If None, uses default.

    Returns:
        img_with_region: Original image with transform region drawn
        warped: Warped (bird's-eye view) image
    """
    if src is None:
        src = get_default_src_points(img.shape)

    if dst is None:
        dst = get_default_dst_points(img.shape)

    # Draw transform region on original
    img_with_region = draw_transform_region(img, src)

    # Get transform matrix
    M, _ = get_perspective_transform(src, dst)

    # Warp image
    warped = warp_image(img, M)

    return img_with_region, warped


def corners_unwarp(
    img: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute perspective transform and return warped image and matrices.

    This is a convenience function that combines multiple operations.

    Args:
        img: Input image
        src: Source points (4, 2) array
        dst: Destination points (4, 2) array

    Returns:
        warped: Warped image
        M: Forward transform matrix
        Minv: Inverse transform matrix
    """
    # Get transform matrices
    M, Minv = get_perspective_transform(src, dst)

    # Warp image
    img_size = (img.shape[1], img.shape[0])
    warped = warp_image(img, M, img_size)

    return warped, M, Minv


if __name__ == "__main__":
    """
    Example usage: Test perspective transform on a test image
    """
    print("=" * 60)
    print("Perspective Transform Test")
    print("=" * 60)

    # Load test image
    img = cv2.imread("test_images/straight_lines1.jpg")

    if img is None:
        print("✗ Could not load test image")
    else:
        print("✓ Loaded test image")

        # Get default transform points
        src = get_default_src_points(img.shape)
        dst = get_default_dst_points(img.shape)

        print("\nSource points (trapezoid on road):")
        for i, pt in enumerate(src):
            print(f"  Point {i+1}: ({pt[0]:.0f}, {pt[1]:.0f})")

        print("\nDestination points (bird's-eye view):")
        for i, pt in enumerate(dst):
            print(f"  Point {i+1}: ({pt[0]:.0f}, {pt[1]:.0f})")

        # Visualize transform
        img_with_region, warped = visualize_transform(img, src, dst)

        # Save results
        cv2.imwrite("docs/perspective_source.jpg", img_with_region)
        cv2.imwrite("docs/perspective_warped.jpg", warped)

        # Create side-by-side comparison
        comparison = np.hstack((img_with_region, warped))
        cv2.imwrite("docs/perspective_comparison.jpg", comparison)

        print("\n✓ Saved perspective transform visualization:")
        print("  - docs/perspective_source.jpg (with region)")
        print("  - docs/perspective_warped.jpg (bird's-eye view)")
        print("  - docs/perspective_comparison.jpg (side-by-side)")

        # Test on straight lines to verify parallel lanes
        straight_img = cv2.imread("test_images/straight_lines2.jpg")
        if straight_img is not None:
            _, warped_straight = visualize_transform(straight_img, src, dst)
            cv2.imwrite("docs/perspective_straight_lines.jpg", warped_straight)
            print("  - docs/perspective_straight_lines.jpg (verify parallel lines)")

        print("\n✓ Perspective transform test complete!")
