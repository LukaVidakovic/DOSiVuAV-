# Advanced Lane Detection

**Computer Vision Project for Autonomous Vehicle Lane Detection**

This project implements a complete lane detection pipeline using computer vision techniques to identify and track lane boundaries on the road.

---

## Project Goals

The objectives of this project are:

* Compute camera calibration matrix and distortion coefficients from chessboard images
* Apply distortion correction to raw road images
* Use color transforms and gradients to create binary thresholded images
* Apply perspective transform to obtain bird's-eye view
* Detect lane pixels and fit polynomial curves
* Calculate lane curvature radius and vehicle position relative to center
* Overlay detected lane boundaries onto original image
* Display curvature and position information

---

## Project Structure

```
.
├── src/                           # Source code modules
│   ├── camera_calibration.py      # Camera calibration and undistortion
│   ├── thresholding.py            # Binary thresholding (gradient + color)
│   ├── perspective_transform.py   # Perspective transformation
│   ├── lane_detection.py          # Lane detection and polynomial fitting
│   └── pipeline.py                # Complete integrated pipeline
├── tests/                         # Unit tests (60 tests, 100% passing)
├── camera_cal/                    # Chessboard calibration images
├── test_images/                   # Test road images
├── test_videos/                   # Test video sequences
├── docs/                          # Documentation images
├── output_images/                 # Processed output images
├── output_videos/                 # Processed output videos
└── README.md                      # This file
```

---

## Camera Calibration

### Computing Camera Matrix and Distortion Coefficients

The camera calibration is implemented in `src/camera_calibration.py` using OpenCV's calibration functions.

**Process:**

1. **Prepare object points**: 3D coordinates (x, y, z) of chessboard corners in real-world space, assuming z=0 (flat plane)
2. **Find chessboard corners**: Use `cv2.findChessboardCorners()` to detect 9x6 inner corners in calibration images
3. **Refine corner positions**: Apply `cv2.cornerSubPix()` for sub-pixel accuracy
4. **Calibrate camera**: Use `cv2.calibrateCamera()` with object points and image points to compute camera matrix and distortion coefficients
5. **Validate**: Calculate mean re-projection error to verify calibration quality

**Code reference:** `src/camera_calibration.py:19-120`

**Results:**
- Successfully calibrated using **17 out of 20** chessboard images
- Mean re-projection error: **0.12 pixels** (excellent accuracy)
- Camera matrix and distortion coefficients saved to `calibration.npz`

**Example:**

![Undistorted Comparison](./docs/undistort_comparison.jpg)

*Left: Original distorted image | Right: Undistorted image*

The distortion correction successfully removes lens distortion, particularly visible at the edges of the image.

---

## Pipeline (Single Images)

### 1. Distortion Correction

All road images are first undistorted using the calibration parameters.

**Code reference:** `src/camera_calibration.py:160-198` - `undistort_image()`

The distortion correction is subtle but important for accurate measurements, especially when calculating curvature and vehicle position.

### 2. Binary Thresholding

I used a combination of gradient and color thresholds to robustly detect lane lines under various lighting conditions.

**Code reference:** `src/thresholding.py:212-271` - `combined_threshold()`

**Gradient Thresholds:**
- **Sobel X gradient** (`abs_sobel_thresh()`): Detects vertical edges (lane lines are mostly vertical)
- **Gradient magnitude** (`mag_thresh()`): Captures overall edge strength
- **Gradient direction** (`dir_thresh()`): Filters edges by orientation

**Color Thresholds:**
- **HLS S-channel** (`hls_select()`): Excellent for detecting both yellow and white lane lines under varying lighting
- **RGB R-channel** (`rgb_select()`): Particularly good for yellow lanes

**Combination Strategy:**
```python
gradient_binary = ((sobel_x == 1) & (magnitude == 1)) & (direction == 1)
color_binary = (hls_s == 1) | (rgb_r == 1)
combined = (gradient_binary == 1) | (color_binary == 1)
```

**Example:**

![Binary Threshold Comparison](./docs/binary_threshold_comparison.jpg)

*Left: Original grayscale | Right: Binary thresholded image*

The combined approach successfully extracts lane features while suppressing noise and irrelevant features.

### 3. Perspective Transform

To simplify lane detection, I transform the image to a bird's-eye view where lane lines appear parallel.

**Code reference:** `src/perspective_transform.py:74-97` - `get_perspective_transform()`, `warp_image()`

**Source points** (trapezoid on road):
```python
src = np.float32([
    [576, 454],   # Top-left
    [704, 454],   # Top-right
    [1088, 684],  # Bottom-right
    [192, 684]    # Bottom-left
])
```

**Destination points** (rectangle for bird's-eye view):
```python
dst = np.float32([
    [320, 0],     # Top-left
    [960, 0],     # Top-right
    [960, 720],   # Bottom-right
    [320, 720]    # Bottom-left
])
```

These points were chosen to:
- Focus on the region ahead where lane lines are visible
- Create sufficient width to accommodate curved lanes
- Ensure parallel lanes in warped view

**Verification:**

![Perspective Transform Comparison](./docs/perspective_comparison.jpg)

*Left: Source image with transform region | Right: Bird's-eye view*

Testing on straight lines confirms the transform is correct:

![Straight Lines Warped](./docs/perspective_straight_lines.jpg)

The parallel lane lines in the bird's-eye view validate the perspective transform.

### 4. Lane Pixel Detection and Polynomial Fitting

**Code reference:** `src/lane_detection.py:57-151` - `find_lane_pixels()`, `fit_polynomial()`

**Sliding Window Search Algorithm:**

1. **Find lane base positions**: Take histogram of bottom half of binary warped image. Peak positions indicate lane x-coordinates
2. **Sliding windows**: Use 9 windows stacked vertically
   - Start from base position
   - Search within ±100 pixel margin
   - If >50 pixels found, recenter window on mean x-position
3. **Extract pixels**: Collect all pixels within windows for left and right lanes
4. **Fit polynomial**: Use `np.polyfit()` to fit 2nd order polynomial: x = Ay² + By + C

**Visualization:**

![Lane Detection Visualization](./docs/lane_detection_visualization.jpg)

*Red pixels: Left lane | Blue pixels: Right lane | Yellow lines: Fitted polynomials*

**Results on test1.jpg:**
- Left lane: 106,032 pixels detected
- Right lane: 31,344 pixels detected
- Polynomial coefficients successfully fitted

### 5. Radius of Curvature and Vehicle Position

**Code reference:** `src/lane_detection.py:233-275` - `calculate_curvature()`, `calculate_vehicle_position()`

**Curvature Calculation:**

The radius of curvature is calculated using the formula:

```
R_curve = ((1 + (2Ay + B)²)^(3/2)) / |2A|
```

where A and B are polynomial coefficients, evaluated at the bottom of the image (closest to vehicle).

**Pixel to Meter Conversion:**
- Y dimension: 30 meters / 720 pixels = 0.0417 m/pixel (dashed line length)
- X dimension: 3.7 meters / 700 pixels = 0.0053 m/pixel (US lane width)

**Vehicle Position:**

Calculated as the difference between lane center and image center (assuming camera mounted at vehicle center):

```python
lane_center = (left_lane_x + right_lane_x) / 2
vehicle_center = image_width / 2
offset = (vehicle_center - lane_center) * xm_per_pix
```

**Example results:**
- test1.jpg: Curvature = 8710m (gentle curve), Offset = -0.26m (left of center)
- test2.jpg: Curvature = 767m (sharp curve), Offset = -0.34m (left of center)
- straight_lines1.jpg: Curvature = 13105m (nearly straight), Offset = -0.12m

### 6. Final Result with Lane Overlay

**Code reference:** `src/pipeline.py:97-189` - `draw_lane()`, `add_text_overlay()`

The final step overlays the detected lane area back onto the original undistorted image:

1. Draw filled polygon between lane lines on blank image (green)
2. Draw lane line boundaries (red for left, blue for right)
3. Unwarp back to original perspective using inverse transform matrix
4. Blend with original image using `cv2.addWeighted()`
5. Add text overlay with curvature and vehicle position

**Example Output:**

![Pipeline Output](./output_images/test1_pipeline_output.jpg)

*Complete pipeline output with lane overlay, curvature (8710m), and vehicle position (-0.26m left)*

---

## Pipeline (Video)

### Video Processing Implementation

**Code reference:** `src/pipeline.py:191-279` - `process_video()`

For video processing, I implemented optimizations to leverage temporal coherence:

1. **State management**: Store previous frame's polynomial coefficients
2. **Optimized search**: Use `search_around_poly()` instead of full sliding window search
   - Search only within ±100 pixels of previous polynomial
   - Much faster than sliding window (2-3x speedup)
3. **Fallback mechanism**: If detection fails, use previous frame's polynomials

### Command-Line Interface

```bash
# Process single image
python3 src/pipeline.py --image test_images/test1.jpg

# Process video
python3 src/pipeline.py --video test_videos/project_video01.mp4 --output output.mp4

# Display during processing
python3 src/pipeline.py --video test_videos/project_video01.mp4 --display
```

### Video Results

*(To be added after processing test videos)*

---

## Discussion

### Implementation Approach

I adopted a modular architecture with separate components for each pipeline stage:

1. **camera_calibration.py**: Isolated calibration logic with save/load functionality
2. **thresholding.py**: Multiple thresholding techniques (gradient, color, combined)
3. **perspective_transform.py**: Flexible transform with configurable points
4. **lane_detection.py**: Core detection algorithms (sliding window, polynomial fitting)
5. **pipeline.py**: Integration layer with state management for video

This modular design provides:
- **Testability**: Each module has comprehensive unit tests (60 tests total)
- **Reusability**: Functions can be used independently
- **Maintainability**: Easy to modify individual components
- **Debuggability**: Each stage can be visualized separately

### Challenges and Solutions

**Challenge 1: Binary Thresholding Robustness**
- **Problem**: Single threshold fails under varying lighting (shadows, bright sections)
- **Solution**: Combine gradient and color thresholds
  - Gradients detect edges regardless of color
  - HLS S-channel works well in shadows
  - RGB R-channel enhances yellow detection

**Challenge 2: Perspective Transform Point Selection**
- **Problem**: Incorrect points lead to non-parallel lanes
- **Solution**: Tested on straight line images to verify parallelism
  - Adjusted points iteratively
  - Used proportional coordinates (works for different resolutions)

**Challenge 3: Handling Missing Lane Markings**
- **Problem**: Worn or missing markings reduce detected pixels
- **Solution**:
  - Lower minpix threshold (50 pixels to recenter window)
  - State persistence for video (use previous frame)
  - Gradient thresholds capture faint markings

### Potential Failure Cases

My pipeline would likely fail or struggle with:

1. **Extreme Lighting Conditions**
   - Very dark roads (night driving without street lights)
   - Direct sun glare washing out lane markings
   - **Improvement**: Adaptive thresholding, histogram equalization

2. **Severe Weather**
   - Heavy rain, snow obscuring lanes
   - Water reflections creating false edges
   - **Improvement**: Temporal filtering, motion-based validation

3. **Road Features**
   - Construction zones with non-standard markings
   - Sharp curves where lanes leave field of view
   - Multiple lanes (highway) causing confusion
   - **Improvement**: Constraint-based lane width validation, road edge detection

4. **Occlusions**
   - Vehicles in adjacent lanes blocking view
   - Shadows from overpasses
   - **Improvement**: Track last known good position, predict trajectory

### Improvements for Production System

1. **Sanity Checks**
   ```python
   # Lane width validation (US: 3.5-4.0m)
   if not 3.0 < lane_width < 4.5:
       reject_detection()

   # Curvature reasonableness
   if curvature < 100:  # < 100m is very sharp
       additional_validation()

   # Parallel lines check
   if abs(left_curve - right_curve) > threshold:
       reject_detection()
   ```

2. **Temporal Smoothing**
   - Average polynomials over last N frames
   - Reduces jitter and wobbly lines
   - Reject outliers using moving average

3. **Confidence Scoring**
   - Number of pixels detected
   - Fit quality (residual error)
   - Consistency with previous frames
   - Use confidence to weight temporal smoothing

4. **Multi-Frame Analysis**
   - Track lanes over multiple frames
   - Use Kalman filtering for prediction
   - Detect and handle lane changes

5. **Adaptive Parameters**
   - Adjust thresholds based on recent success rate
   - Expand search area if detection failing
   - Dynamic window size based on curvature

---

## Installation and Usage

### Prerequisites

- Python 3.8+
- OpenCV 4.5+
- NumPy
- Matplotlib
- pytest (for testing)

### Setup

```bash
# Clone repository
git clone https://github.com/LukaVidakovic/DOSiVuAV-.git
cd DOSiVuAV-

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# 1. Camera Calibration (one-time setup)
python3 src/camera_calibration.py
# Creates calibration.npz

# 2. Process single image
python3 src/pipeline.py --image test_images/test1.jpg

# 3. Process video
python3 src/pipeline.py --video test_videos/project_video01.mp4
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_camera_calibration.py -v
pytest tests/test_thresholding.py -v
pytest tests/test_perspective_transform.py -v
pytest tests/test_lane_detection.py -v
pytest tests/test_pipeline.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Test Results

✅ **60 out of 60 unit tests passing** (100% success rate)

- camera_calibration: 4/4 tests ✅
- thresholding: 12/12 tests ✅
- perspective_transform: 14/14 tests ✅
- lane_detection: 15/15 tests ✅
- pipeline: 15/15 tests ✅

---

## References

- OpenCV Camera Calibration: [docs.opencv.org](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)
- Sobel Edge Detection: [docs.opencv.org](https://docs.opencv.org/master/d2/d2c/tutorial_sobel_derivatives.html)
- Perspective Transform: [docs.opencv.org](https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html)

---

## Author

Luka Vidakovic

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
