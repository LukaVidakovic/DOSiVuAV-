# Advanced Lane Finding Project

**Computer Vision for Autonomous Vehicles**

This project implements a complete lane detection pipeline for autonomous vehicle applications using OpenCV in Python. The system detects lane lines from camera images/videos, calculates curvature and vehicle position, achieving high accuracy (>95%) on test datasets.

---

## Project Goals

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply a distortion correction to raw images
* Use color transforms and morphological operations to create a thresholded binary image
* Apply a perspective transform to rectify binary image ("birds-eye view")
* Detect lane pixels using sliding window technique
* Fit polynomial curves to find the lane boundaries
* Determine the curvature of the lane and vehicle position with respect to center
* Warp the detected lane boundaries back onto the original image
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

---

## Project Structure

```
├── calibrate_camera.py          # Camera calibration script
├── lane_detection.py            # Main lane detection pipeline
├── generate_docs_images.py      # Documentation image generator
├── calibration.npz              # Camera calibration parameters
├── camera_cal/                  # Chessboard calibration images
├── test_images/                 # Test images
├── test_videos/                 # Test videos
├── output_images/               # Processed images
├── output_videos/               # Processed videos
├── docs/                        # Documentation images
├── debug/                       # Development and debugging scripts
└── README.md                    # This file
```

---

## Pipeline Overview

The lane detection pipeline consists of 8 main steps:

1. **Camera Calibration** - Compute camera matrix and distortion coefficients
2. **Distortion Correction** - Undistort raw images
3. **Binary Thresholding** - Create binary image using HSV color space
4. **Perspective Transform** - Warp to bird's-eye view
5. **Lane Pixel Detection** - Find lane pixels using sliding windows
6. **Polynomial Fitting** - Fit 2nd order polynomial to lane lines
7. **Curvature Calculation** - Calculate radius and vehicle position
8. **Visualization** - Draw lane area back onto original image

---

## Detailed Pipeline Steps

### 1. Camera Calibration

**Code:** `calibrate_camera.py`

Camera calibration is essential to correct lens distortion. I used 20 chessboard images (9x6 corners) provided in the `camera_cal/` folder.

**Process:**
- Prepare object points in 3D space (x, y, z) where z=0
- Find chessboard corners in each calibration image using `cv2.findChessboardCorners()`
- Collect object points and image points from successful detections
- Use `cv2.calibrateCamera()` to compute camera matrix and distortion coefficients
- Save parameters to `calibration.npz` for reuse

**Results:**
- Successfully processed: **17/20 images**
- 3 images failed (calibration1.jpg, calibration4.jpg, calibration5.jpg) due to partial chessboard visibility

![Calibration Example](./docs/calibration_example.jpg)
*Left: Original distorted image | Right: Undistorted image*

**Key Code:**
```python
def calibrate_camera(images_path='camera_cal/calibration*.jpg', nx=9, ny=6):
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    
    for fname in glob.glob(images_path):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    return mtx, dist
```

---

### 2. Distortion Correction

**Code:** `lane_detection.py` - `process_image()` function

Using the calibration parameters from step 1, I undistort each image using `cv2.undistort()`.

![Original](./docs/01_original.jpg)
*Original test image*

![Undistorted](./docs/02_undistorted.jpg)
*Undistorted image - notice subtle correction at edges*

**Key Code:**
```python
def load_calibration(calib_file='calibration.npz'):
    data = np.load(calib_file)
    return data['mtx'], data['dist']

# In pipeline:
undist = cv2.undistort(img, mtx, dist, None, mtx)
```

---

### 3. Binary Thresholding

**Code:** `lane_detection.py` - `combined_threshold()` function

This is one of the most critical steps. After extensive testing with different color spaces (HLS, LAB, RGB), I found that **HSV color space** provides the best results for detecting both yellow and white lane lines.

**Approach:**
1. Convert image to HSV color space
2. Apply color masks:
   - **Yellow lanes**: H=15-40, S=80-255, V=160-255
   - **White lanes**: H=0-255, S=0-20, V=200-255
3. Combine masks with bitwise OR
4. Apply morphological closing (5x5 kernel, 3 iterations) to remove noise

**Why HSV over HLS?**
- HSV is more robust to lighting variations
- Better separation of yellow and white colors
- Improved performance on challenge videos (95.7% vs 19.6% with HLS)

![Binary Threshold](./docs/03_binary_threshold.jpg)
*Binary thresholded image - white pixels represent detected lane lines*

**Key Code:**
```python
def combined_threshold(img):
    """HSV color space with morphological closing"""
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
```

---

### 4. Perspective Transform

**Code:** `lane_detection.py` - `get_perspective_transform()` function

To accurately detect curved lanes, I transform the image to a bird's-eye view. This makes parallel lane lines appear parallel in the transformed image.

**Source Points (trapezoid in original image):**
- Top-left: (w*0.43, h*0.65)
- Top-right: (w*0.57, h*0.65)
- Bottom-left: (w*0.20, h*0.95)
- Bottom-right: (w*0.80, h*0.95)

**Destination Points (rectangle in warped image):**
- Top-left: (w*0.25, 0)
- Top-right: (w*0.75, 0)
- Bottom-left: (w*0.25, h)
- Bottom-right: (w*0.75, h)

![Perspective Source](./docs/04_perspective_src.jpg)
*Source trapezoid (blue lines) defining region of interest*

![Warped](./docs/05_warped.jpg)
*Bird's-eye view after perspective transform*

**Key Code:**
```python
def get_perspective_transform(img_shape):
    h, w = img_shape[:2]
    
    src = np.float32([
        [w*0.43, h*0.65],  # Top-left
        [w*0.57, h*0.65],  # Top-right
        [w*0.20, h*0.95],  # Bottom-left
        [w*0.80, h*0.95]   # Bottom-right
    ])
    
    dst = np.float32([
        [w*0.25, 0],       # Top-left
        [w*0.75, 0],       # Top-right
        [w*0.25, h],       # Bottom-left
        [w*0.75, h]        # Bottom-right
    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return M, Minv
```

---

### 5. Lane Pixel Detection (Sliding Windows)

**Code:** `lane_detection.py` - `find_lane_pixels()` function

I use the sliding window technique to identify which pixels belong to left and right lane lines.

**Algorithm:**
1. Calculate histogram of bottom half of image
2. Find peaks in left and right halves (starting points)
3. Divide image into 9 horizontal windows
4. For each window:
   - Define window boundaries (±100 pixels margin)
   - Identify non-zero pixels within window
   - If enough pixels found (>50), recenter next window
5. Collect all identified lane pixels

![Sliding Windows](./docs/06_sliding_windows.jpg)
*Sliding windows (green rectangles) tracking lane pixels (red=left, blue=right)*

**Key Code:**
```python
def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    # Histogram to find starting points
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        # Window boundaries
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify pixels within window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Recenter if enough pixels found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty
```

---

### 6. Polynomial Fitting

**Code:** `lane_detection.py` - `fit_polynomial()` function

I fit a 2nd order polynomial to the detected lane pixels:

**f(y) = Ay² + By + C**

Why fit to y instead of x? Because lane lines can be vertical, making x as a function of y more stable.

![Polynomial Fit](./docs/07_polynomial_fit.jpg)
*Yellow curves show fitted 2nd order polynomials*

**Key Code:**
```python
def fit_polynomial(leftx, lefty, rightx, righty):
    """Fit 2nd order polynomial"""
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit
```

---

### 7. Curvature and Vehicle Position

**Code:** `lane_detection.py` - `calculate_curvature()` and `calculate_vehicle_position()` functions

#### Radius of Curvature

The radius of curvature is calculated using the formula:

**R = [(1 + (2Ay + B)²)^(3/2)] / |2A|**

Where A and B are polynomial coefficients in world space (meters).

**Pixel to Meter Conversion:**
- Y-axis: 30 meters / 720 pixels
- X-axis: 3.7 meters / 700 pixels (US highway lane width)

**Temporal Smoothing:**
To reduce jitter in videos, I average the current radius with the previous frame's radius.

#### Vehicle Position

Calculate the offset from lane center:
1. Find lane center: (left_lane_bottom + right_lane_bottom) / 2
2. Find image center: image_width / 2
3. Offset = (lane_center - image_center) * xm_per_pix

Negative offset = vehicle is left of center  
Positive offset = vehicle is right of center

**Key Code:**
```python
# Conversion factors
YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

def calculate_curvature(left_fit, right_fit, y_eval, img_shape):
    """Calculate radius with temporal smoothing"""
    global last_radius
    
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Fit polynomial in world space
    left_fit_cr = np.polyfit(ploty*YM_PER_PIX, left_fitx*XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(ploty*YM_PER_PIX, right_fitx*XM_PER_PIX, 2)
    
    y_eval_world = y_eval * YM_PER_PIX
    
    # Calculate radius
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_world + left_fit_cr[1])**2)**1.5) / np.abs(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_world + right_fit_cr[1])**2)**1.5) / np.abs(2*right_fit_cr[0])
    
    # Temporal smoothing
    current = np.mean([left_curverad, right_curverad])
    if last_radius is not None:
        smoothed = np.mean([current, last_radius])
    else:
        smoothed = current
    last_radius = current
    
    return smoothed, smoothed

def calculate_vehicle_position(left_fit, right_fit, img_width, img_height):
    """Calculate vehicle offset from lane center"""
    y_eval = img_height - 1
    
    left_lane_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_lane_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    lane_center = (left_lane_bottom + right_lane_bottom) / 2
    img_center = img_width / 2
    
    offset = (img_center - lane_center) * XM_PER_PIX
    
    return offset
```

---

### 8. Visualization

**Code:** `lane_detection.py` - `draw_lane()` and `add_text()` functions

Final step is to visualize the detected lane:

1. Create blank image same size as warped image
2. Draw filled polygon between left and right lane lines (green)
3. Draw lane lines (red=left, blue=right, 15px thickness)
4. Warp back to original perspective using inverse matrix (Minv)
5. Overlay on original undistorted image with transparency
6. Add text showing curvature and vehicle position

![Final Result](./docs/08_final_result.jpg)
*Final result with detected lane area and metrics*

**Key Code:**
```python
def draw_lane(undist, binary_warped, left_fit, right_fit, Minv):
    """Draw detected lane area"""
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create blank image
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Fill lane area (green)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Draw lane lines
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    right_line = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
    cv2.polylines(color_warp, left_line, False, (255, 0, 0), 15)
    cv2.polylines(color_warp, right_line, False, (0, 0, 255), 15)
    
    # Warp back to original perspective
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    
    # Overlay on original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

def add_text(img, curvature, offset):
    """Add curvature and position text"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Curvature text
    curv_text = f'Radius of Curvature: {int(curvature)}m'
    cv2.putText(img, curv_text, (50, 50), font, 1.2, (255, 255, 255), 2)
    
    # Position text
    direction = "left" if offset < 0 else "right"
    pos_text = f'Vehicle is {abs(offset):.2f}m {direction} of center'
    cv2.putText(img, pos_text, (50, 100), font, 1.2, (255, 255, 255), 2)
    
    return img
```

---

## Results

### Test Images

Successfully processed **17/17 test images (100%)**:

| Image | Curvature | Offset | Status |
|-------|-----------|--------|--------|
| challange00101.jpg | 824m | -0.23m | ✓ |
| challange00111.jpg | 1945m | -0.15m | ✓ |
| challange00136.jpg | 868m | -0.01m | ✓ |
| solidWhiteCurve.jpg | 7546m | -0.20m | ✓ |
| solidWhiteRight.jpg | 8334m | -0.03m | ✓ |
| solidYellowCurve.jpg | 1404m | -0.09m | ✓ |
| solidYellowCurve2.jpg | 10403m | -0.10m | ✓ |
| solidYellowLeft.jpg | 2630m | -0.04m | ✓ |
| straight_lines1.jpg | 6509m | -0.05m | ✓ |
| straight_lines2.jpg | 11867m | -0.09m | ✓ |
| test1.jpg | 10503m | -0.24m | ✓ |
| test2.jpg | 4532m | -0.44m | ✓ |
| test3.jpg | 2894m | -0.19m | ✓ |
| test4.jpg | 2756m | -0.37m | ✓ |
| test5.jpg | 1293m | -0.06m | ✓ |
| test6.jpg | 2272m | -0.32m | ✓ |
| whiteCarLaneSwitch.jpg | 4298m | -0.17m | ✓ |

### Test Videos

| Video | Frames | Success Rate | Notes |
|-------|--------|--------------|-------|
| project_video01.mp4 | 221/221 | **100.0%** | Perfect detection |
| project_video02.mp4 | 681/681 | **100.0%** | Perfect detection |
| project_video03.mp4 | 1260/1260 | **100.0%** | Perfect detection |
| challenge01.mp4 | 251/251 | **100.0%** | Excellent |
| challenge02.mp4 | 463/484 | **95.7%** | Very challenging lighting |
| challenge03.mp4 | 1079/1199 | **90.0%** | Sharp curves, shadows |

**Overall Video Success Rate: 97.8%**

---

## Key Improvements and Techniques

### 1. HSV Color Space
After testing HLS, LAB, and RGB color spaces, **HSV proved most robust**:
- Better yellow/white separation
- More resistant to lighting changes
- Improved challenge video performance from 19.6% to 95.7%

### 2. Morphological Closing
Applying morphological closing (5x5 kernel, 3 iterations) significantly reduced noise:
- Removed small gaps in lane lines
- Connected broken line segments
- Reduced false positives from road texture

### 3. Temporal Smoothing
Averaging current radius with previous frame's radius:
- Reduced jitter in video output
- More stable curvature readings
- Better user experience

### 4. Adaptive Perspective Transform
Using relative coordinates (percentages of image dimensions):
- Works with different resolutions (720p, 540p)
- Maintains correct aspect ratio
- No hardcoded pixel values

---

## Discussion

### Challenges Faced

1. **Color Space Selection**
   - Initially used HLS which worked well for simple cases
   - Failed on challenge videos with varying lighting
   - Solution: Switched to HSV with carefully tuned thresholds

2. **Noise Reduction**
   - Binary threshold produced ~400k pixels with too much noise
   - Solution: Morphological closing reduced noise by 96%

3. **Line Alignment**
   - Overlay lines didn't match actual lanes precisely
   - Solution: Reduced line thickness from 25px to 15px

4. **Video Stability**
   - Curvature values jumped between frames
   - Solution: Temporal smoothing with previous frame

### Pipeline Limitations

1. **Sharp Curves**
   - Performance degrades on very sharp turns (challenge03: 90%)
   - 2nd order polynomial may not be sufficient for extreme curves
   - Potential solution: Higher order polynomial or spline fitting

2. **Extreme Lighting**
   - Struggles with very bright or very dark conditions
   - Challenge02 has 95.7% success due to lighting variations
   - Potential solution: Adaptive histogram equalization

3. **Road Markings**
   - Can be confused by other road markings (arrows, text)
   - Potential solution: Add sanity checks for lane width and parallelism

4. **Weather Conditions**
   - Not tested on rain, snow, or fog
   - Potential solution: Additional preprocessing for weather conditions

### Future Improvements

1. **Sanity Checks**
   - Verify lane width is reasonable (3-4 meters)
   - Check that lanes are roughly parallel
   - Reject detections with high polynomial fit residuals

2. **Temporal Tracking**
   - Use previous frame's polynomial as starting point
   - Skip sliding windows when lanes are already detected
   - Implement Kalman filter for smoother tracking

3. **Deep Learning**
   - Use CNN for semantic segmentation
   - More robust to varying conditions
   - Can handle complex scenarios

4. **Multi-frame Averaging**
   - Average polynomial coefficients over multiple frames
   - Further reduce jitter and noise
   - More stable curvature estimation

---

## How to Run

### Prerequisites

```bash
pip install numpy opencv-python matplotlib
```

### Step 1: Camera Calibration

```bash
python calibrate_camera.py
```

This generates `calibration.npz` file.

### Step 2: Process Images

```bash
python lane_detection.py --images
```

Results saved to `output_images/` folder.

### Step 3: Process Videos

```bash
python lane_detection.py --videos
```

Results saved to `output_videos/` folder.

### Step 4: Generate Documentation Images

```bash
python generate_docs_images.py
```

Pipeline step images saved to `docs/` folder.

---

## Author

**Luka Vidaković**  
University of Novi Sad  
Faculty of Technical Sciences  
E2 40/2025

---

## License

This project is licensed under the MIT License.
