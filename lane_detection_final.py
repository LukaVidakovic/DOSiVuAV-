#!/usr/bin/env python3
"""
Finalni Lane Detection Pipeline - Čist i Funkcionalan

Integriše sve testirane komponente u jedan pipeline.
"""

import numpy as np
import cv2
import os
import glob

# Global for temporal smoothing
last_radius = None

# ============================================================================
# 1. KALIBRACIJA KAMERE
# ============================================================================

def load_calibration(calib_file='calibration.npz'):
    """Load camera calibration"""
    data = np.load(calib_file)
    return data['mtx'], data['dist']

# ============================================================================
# 2. BINARNA SEGMENTACIJA
# ============================================================================

def combined_threshold(img):
    """HSV color space with morphological closing"""
    
    # HSV color space (better for yellow/white than HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Yellow: H=15-40, S=80-255, V=160-255
    yellow = cv2.inRange(hsv, (15, 80, 160), (40, 255, 255))
    
    # White: H=0-255, S=0-20, V=200-255
    white = cv2.inRange(hsv, (0, 0, 200), (255, 20, 255))
    
    # Combine masks
    combined = cv2.bitwise_or(yellow, white)
    
    # Morphological closing to remove noise
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return combined // 255

# ============================================================================
# 3. PERSPEKTIVNA TRANSFORMACIJA
# ============================================================================

def get_perspective_transform(img_shape):
    """Get adaptive perspective transform - narrower to avoid barriers"""
    h, w = img_shape[:2]
    
    src = np.float32([
        [w * 0.43, h * 0.65],
        [w * 0.57, h * 0.65],
        [w * 0.20, h * 0.95],
        [w * 0.80, h * 0.95]
    ])
    
    dst = np.float32([
        [w * 0.25, 0],
        [w * 0.75, 0],
        [w * 0.25, h],
        [w * 0.75, h]
    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return M, Minv

# ============================================================================
# 4. DETEKCIJA PIKSELA LINIJA
# ============================================================================

def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    """Sliding window search for lane pixels"""
    # Histogram
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Window setup
    window_height = binary_warped.shape[0] // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    # Slide windows
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty

# ============================================================================
# 5. FITOVANJE POLINOMA
# ============================================================================

def fit_polynomial(leftx, lefty, rightx, righty):
    """Fit 2nd order polynomial"""
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

# ============================================================================
# 6. ZAKRIVLJENOST I POZICIJA
# ============================================================================

YM_PER_PIX = 30/720
XM_PER_PIX = 3.7/700

def calculate_curvature(left_fit, right_fit, y_eval, img_shape):
    """Calculate radius with temporal smoothing"""
    global last_radius
    
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    left_fit_cr = np.polyfit(ploty*YM_PER_PIX, left_fitx*XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(ploty*YM_PER_PIX, right_fitx*XM_PER_PIX, 2)
    
    y_eval_world = y_eval * YM_PER_PIX
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_world + left_fit_cr[1])**2)**1.5) / np.abs(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_world + right_fit_cr[1])**2)**1.5) / np.abs(2*right_fit_cr[0])
    
    # Temporal smoothing for video stability
    current = np.mean([left_curverad, right_curverad])
    if last_radius is not None:
        smoothed = np.mean([current, last_radius])
    else:
        smoothed = current
    last_radius = current
    
    return smoothed, smoothed

def calculate_vehicle_position(left_fit, right_fit, img_width, img_height):
    """Calculate vehicle offset from center"""
    y_max = img_height - 1
    left_lane_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    right_lane_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
    lane_center = (left_lane_pos + right_lane_pos) / 2
    vehicle_center = img_width / 2
    offset = (vehicle_center - lane_center) * XM_PER_PIX
    return offset

# ============================================================================
# 7. VIZUALIZACIJA
# ============================================================================

def draw_lane(undist, binary_warped, left_fit, right_fit, Minv):
    """Draw lane overlay on original image"""
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), thickness=15)
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

def add_text(img, curvature, offset):
    """Add text overlay"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if curvature > 10000:
        curv_text = "Radius of Curvature: Straight"
    else:
        curv_text = f"Radius of Curvature: {curvature:.0f}m"
    
    if abs(offset) < 0.05:
        pos_text = "Vehicle Position: Centered"
    elif offset < 0:
        pos_text = f"Vehicle Position: {abs(offset):.2f}m left of center"
    else:
        pos_text = f"Vehicle Position: {offset:.2f}m right of center"
    
    cv2.putText(img, curv_text, (50, 50), font, 1.0, (255, 255, 255), 2)
    cv2.putText(img, pos_text, (50, 100), font, 1.0, (255, 255, 255), 2)
    
    return img

# ============================================================================
# 8. GLAVNI PIPELINE
# ============================================================================

def process_image(img, mtx, dist):
    """Process single image through complete pipeline"""
    
    # 1. Undistort
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # 2. Binary threshold
    binary = combined_threshold(undist)
    
    # 3. Perspective transform
    M, Minv = get_perspective_transform(img.shape)
    binary_warped = cv2.warpPerspective(binary, M, (img.shape[1], img.shape[0]))
    
    # 4. Find lane pixels
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    
    # Check if enough pixels detected
    if len(leftx) < 100 or len(rightx) < 100:
        return None, {"detected": False}
    
    # 5. Fit polynomial
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
    
    # 6. Calculate metrics
    y_eval = binary_warped.shape[0] - 1
    left_curv, right_curv = calculate_curvature(left_fit, right_fit, y_eval, binary_warped.shape)
    avg_curv = (left_curv + right_curv) / 2
    offset = calculate_vehicle_position(left_fit, right_fit, img.shape[1], img.shape[0])
    
    # 7. Draw lane
    result = draw_lane(undist, binary_warped, left_fit, right_fit, Minv)
    result = add_text(result, avg_curv, offset)
    
    stats = {
        "detected": True,
        "curvature": avg_curv,
        "offset": offset,
        "left_pixels": len(leftx),
        "right_pixels": len(rightx)
    }
    
    return result, stats

# ============================================================================
# 9. VIDEO PROCESSING
# ============================================================================

def process_video(video_path, output_path, mtx, dist):
    """Process video file"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Cannot open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    success_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        result, stats = process_image(frame, mtx, dist)
        
        if stats["detected"]:
            out.write(result)
            success_count += 1
        else:
            out.write(frame)  # Write original if detection fails
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')
    
    cap.release()
    out.release()
    
    print(f"\n  ✓ Processed {success_count}/{frame_count} frames ({success_count/frame_count*100:.1f}%)")
    return True

def process_all_videos():
    """Process all test videos"""
    
    print("\n" + "="*60)
    print("VIDEO PROCESSING")
    print("="*60)
    
    mtx, dist = load_calibration()
    
    video_paths = sorted(glob.glob("test_videos/*.mp4"))
    print(f"✓ Found {len(video_paths)} test videos\n")
    
    os.makedirs("output_videos", exist_ok=True)
    
    for video_path in video_paths:
        basename = os.path.basename(video_path)
        output_path = f"output_videos/{basename}"
        
        print(f"Processing: {basename}")
        process_video(video_path, output_path, mtx, dist)
    
    print(f"\n{'='*60}")
    print(f"✓ Videos saved to output_videos/")
    print("="*60)

def process_all_images():
    """Process all test images"""
    
    print("="*60)
    print("FINALNI LANE DETECTION PIPELINE")
    print("="*60)
    
    # Load calibration
    mtx, dist = load_calibration()
    print("✓ Loaded calibration")
    
    # Get all test images
    image_paths = sorted(glob.glob("test_images/*.jpg"))
    print(f"✓ Found {len(image_paths)} test images\n")
    
    # Create output directory
    os.makedirs("output_images", exist_ok=True)
    
    # Process each image
    successes = 0
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        print(f"Processing: {basename:30s} ", end="")
        
        img = cv2.imread(img_path)
        result, stats = process_image(img, mtx, dist)
        
        if stats["detected"]:
            output_path = f"output_images/{basename}"
            cv2.imwrite(output_path, result)
            print(f"✓ Curv: {stats['curvature']:6.0f}m  Offset: {stats['offset']:+.2f}m")
            successes += 1
        else:
            print("✗ Detection failed")
    
    print(f"\n{'='*60}")
    print(f"Success: {successes}/{len(image_paths)} images ({successes/len(image_paths)*100:.1f}%)")
    print("="*60)
    print(f"\n✓ Results saved to output_images/")

if __name__ == "__main__":
    import sys
    
    if "--videos" in sys.argv:
        process_all_videos()
    elif "--images" in sys.argv:
        process_all_images()
    else:
        # Default: process both
        process_all_images()
        
        print("\n" + "="*60)
        response = input("Process videos? (y/n): ")
        if response.lower() == 'y':
            process_all_videos()
