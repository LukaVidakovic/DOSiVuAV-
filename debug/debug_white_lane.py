"""
Debug white lane detection
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '../src')

from camera_calibration import load_calibration, undistort_image
from thresholding import enhance_image, abs_sobel_thresh
from perspective_transform import get_default_src_points, get_default_dst_points, get_perspective_transform, warp_image

# Load calibration
mtx, dist, _, _ = load_calibration('../calibration.npz')

for img_name in ['challange00101', 'test1']:
    print(f"\n{'='*60}")
    print(f"Analyzing: {img_name}")
    print('='*60)

    img = cv2.imread(f'../test_images/{img_name}.jpg')
    undist = undistort_image(img, mtx, dist)
    enhanced = enhance_image(undist, clip_limit=2.5)

    # Convert to color spaces
    hls = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # WHITE detection (current)
    white_binary = np.zeros_like(gray)
    white_binary[(hls[:, :, 1] > 200) & (hls[:, :, 2] < 30)] = 1

    # WHITE detection (stricter lightness)
    white_strict = np.zeros_like(gray)
    white_strict[(hls[:, :, 1] > 220) & (hls[:, :, 2] < 30)] = 1

    # YELLOW detection (current relaxed)
    yellow_binary = np.zeros_like(gray)
    yellow_binary[
        (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 35) &
        (hsv[:, :, 1] >= 50) &
        (hsv[:, :, 2] >= 50)
    ] = 1

    # GRADIENT
    gradx = abs_sobel_thresh(enhanced, orient='x', sobel_kernel=3, thresh=(20, 100))

    print(f"White (L>200): {np.sum(white_binary)}")
    print(f"White (L>220): {np.sum(white_strict)}")
    print(f"Yellow: {np.sum(yellow_binary)}")
    print(f"Gradient: {np.sum(gradx)}")

    # Perspective transform
    src = get_default_src_points(img.shape)
    dst = get_default_dst_points(img.shape)
    M, Minv = get_perspective_transform(src, dst)

    # Warp channels separately
    white_warped = warp_image(white_binary, M)
    white_strict_warped = warp_image(white_strict, M)
    yellow_warped = warp_image(yellow_binary, M)
    gradx_warped = warp_image(gradx, M)

    # Create color visualization showing channels
    # RED = gradient, GREEN = yellow, BLUE = white
    color_viz = np.dstack((gradx_warped*255, yellow_warped*255, white_warped*255)).astype(np.uint8)

    cv2.imwrite(f'../output_images/{img_name}_channels_warped.jpg', color_viz)
    print(f"✓ Saved {img_name}_channels_warped.jpg (RED=grad, GREEN=yellow, BLUE=white)")

    # Check histogram for right lane
    histogram = np.sum(white_warped[white_warped.shape[0]//2:, :], axis=0)
    midpoint = len(histogram) // 2
    right_peak = np.argmax(histogram[midpoint:]) + midpoint
    print(f"Right lane white peak at x={right_peak}")
