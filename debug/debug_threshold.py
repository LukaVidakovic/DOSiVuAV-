"""
Debug thresholding - visualize what pixels are detected
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '../src')

from camera_calibration import load_calibration, undistort_image
from thresholding import combined_threshold, abs_sobel_thresh
from perspective_transform import get_default_src_points, get_default_dst_points, get_perspective_transform, warp_image

# Load calibration
mtx, dist, _, _ = load_calibration('../calibration.npz')

# Test each challenge image
for img_name in ['challange00101', 'challange00111', 'challange00136']:
    print(f"\n{'='*60}")
    print(f"Processing: {img_name}")
    print('='*60)

    img = cv2.imread(f'../test_images/{img_name}.jpg')
    undist = undistort_image(img, mtx, dist)

    # Get binary threshold
    binary = combined_threshold(undist, use_clahe=False)

    # Separate color channel analysis
    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # WHITE detection
    white_binary = np.zeros_like(gray)
    white_binary[(hls[:, :, 1] > 200) & (hls[:, :, 2] < 30)] = 1

    # YELLOW detection
    yellow_binary = np.zeros_like(gray)
    yellow_binary[
        (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 30) &
        (hsv[:, :, 1] >= 70) &
        (hsv[:, :, 2] >= 70)
    ] = 1

    # GRADIENT
    gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=3, thresh=(20, 100))

    print(f"White pixels: {np.sum(white_binary)} ({100*np.sum(white_binary)/white_binary.size:.2f}%)")
    print(f"Yellow pixels: {np.sum(yellow_binary)} ({100*np.sum(yellow_binary)/yellow_binary.size:.2f}%)")
    print(f"Gradient pixels: {np.sum(gradx)} ({100*np.sum(gradx)/gradx.size:.2f}%)")
    print(f"Combined pixels: {np.sum(binary)} ({100*np.sum(binary)/binary.size:.2f}%)")

    # Create visualization
    color_binary = np.dstack((gradx*255, yellow_binary*255, white_binary*255)).astype(np.uint8)

    # Perspective transform
    src = get_default_src_points(img.shape)
    dst = get_default_dst_points(img.shape)
    M, Minv = get_perspective_transform(src, dst)

    binary_warped = warp_image(binary, M)
    color_warped = warp_image(color_binary, M)

    # Save visualizations
    cv2.imwrite(f'../output_images/{img_name}_binary.jpg', binary * 255)
    cv2.imwrite(f'../output_images/{img_name}_color_channels.jpg', color_binary)
    cv2.imwrite(f'../output_images/{img_name}_warped_binary.jpg', binary_warped * 255)
    cv2.imwrite(f'../output_images/{img_name}_warped_color.jpg', color_warped)

    print(f"✓ Saved debug images for {img_name}")
