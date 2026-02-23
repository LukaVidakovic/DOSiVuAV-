"""
Analyze what pixels are detected for each challenge image
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '../src')

from camera_calibration import load_calibration, undistort_image
from thresholding import abs_sobel_thresh, enhance_image
from perspective_transform import get_default_src_points, get_default_dst_points, get_perspective_transform, warp_image
from lane_detection import find_lane_pixels

# Load calibration
mtx, dist, _, _ = load_calibration('../calibration.npz')

for img_name in ['challange00101', 'test1']:
    print(f"\n{'='*60}")
    print(f"Analyzing: {img_name}")
    print('='*60)

    img = cv2.imread(f'../test_images/{img_name}.jpg')
    undist = undistort_image(img, mtx, dist)

    # Try WITH CLAHE
    enhanced = enhance_image(undist, clip_limit=2.5)

    # Convert to color spaces
    hls = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # WHITE detection
    white_binary = np.zeros_like(gray)
    white_binary[(hls[:, :, 1] > 200) & (hls[:, :, 2] < 30)] = 1

    # YELLOW detection (current)
    yellow_binary = np.zeros_like(gray)
    yellow_binary[
        (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 30) &
        (hsv[:, :, 1] >= 80) &
        (hsv[:, :, 2] >= 80)
    ] = 1

    # YELLOW detection (relaxed for faded lines)
    yellow_relaxed = np.zeros_like(gray)
    yellow_relaxed[
        (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 35) &
        (hsv[:, :, 1] >= 50) &
        (hsv[:, :, 2] >= 50)
    ] = 1

    # GRADIENT
    gradx = abs_sobel_thresh(enhanced, orient='x', sobel_kernel=3, thresh=(20, 100))

    print(f"White pixels: {np.sum(white_binary)}")
    print(f"Yellow (current): {np.sum(yellow_binary)}")
    print(f"Yellow (relaxed): {np.sum(yellow_relaxed)}")
    print(f"Gradient: {np.sum(gradx)}")

    # Perspective transform
    src = get_default_src_points(img.shape)
    dst = get_default_dst_points(img.shape)
    M, Minv = get_perspective_transform(src, dst)

    # Create combined versions
    combined_current = np.zeros_like(gray)
    combined_current[(white_binary == 1) | (yellow_binary == 1) | (gradx == 1)] = 1

    combined_relaxed = np.zeros_like(gray)
    combined_relaxed[(white_binary == 1) | (yellow_relaxed == 1) | (gradx == 1)] = 1

    # Warp both
    warped_current = warp_image(combined_current, M)
    warped_relaxed = warp_image(combined_relaxed, M)

    # Find lane pixels with both
    print("\nCurrent thresholds:")
    leftx, lefty, rightx, righty = find_lane_pixels(warped_current)
    print(f"  Left pixels: {len(leftx)}, Right pixels: {len(rightx)}")

    print("Relaxed thresholds:")
    leftx_r, lefty_r, rightx_r, righty_r = find_lane_pixels(warped_relaxed)
    print(f"  Left pixels: {len(leftx_r)}, Right pixels: {len(rightx_r)}")

    # Save comparison
    cv2.imwrite(f'../output_images/{img_name}_warped_current.jpg', warped_current * 255)
    cv2.imwrite(f'../output_images/{img_name}_warped_relaxed.jpg', warped_relaxed * 255)

    # Create color visualization
    color_current = np.dstack((gradx*255, yellow_binary*255, white_binary*255)).astype(np.uint8)
    color_relaxed = np.dstack((gradx*255, yellow_relaxed*255, white_binary*255)).astype(np.uint8)

    cv2.imwrite(f'../output_images/{img_name}_color_current.jpg', color_current)
    cv2.imwrite(f'../output_images/{img_name}_color_relaxed.jpg', color_relaxed)

    print(f"✓ Saved analysis for {img_name}")
