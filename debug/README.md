# Debug Scripts

This folder contains debug scripts for analyzing and visualizing lane detection pipeline.

## Prerequisites

Activate virtual environment from project root:
```bash
cd /Users/lukavidakovic/github/personal/DOSiVuAV-
source venv/bin/activate
```

## Scripts

### 1. debug_pipeline.py
Compare detection WITH and WITHOUT CLAHE enhancement.

**Usage:**
```bash
cd debug
python debug_pipeline.py
```

**Output:**
- Prints pixel counts and polynomial fits
- Saves comparison images in debug/ folder

---

### 2. debug_threshold.py
Visualize color channel detection (white, yellow, gradient).

**Usage:**
```bash
cd debug
python debug_threshold.py
```

**Output:**
- Analyzes challange00101, challange00111, challange00136
- Saves binary and color channel visualizations to ../output_images/

---

### 3. analyze_detection.py
Compare current vs relaxed yellow thresholds.

**Usage:**
```bash
cd debug
python analyze_detection.py
```

**Output:**
- Compares detection on challange00101 and test1
- Saves warped and color visualizations to ../output_images/

---

### 4. debug_white_lane.py
Debug white lane detection specifically.

**Usage:**
```bash
cd debug
python debug_white_lane.py
```

**Output:**
- Analyzes white detection methods
- Creates color-coded channel visualization (RED=gradient, GREEN=yellow, BLUE=white)
- Saves to ../output_images/

---

### 5. test_thresholds.py
Test individual threshold components (gradients, colors).

**Usage:**
```bash
cd debug
python test_thresholds.py
```

**Output:**
- Tests Sobel X, magnitude, direction, HLS, RGB thresholds
- Prints pixel counts for each method
- Saves debug images to ../docs/debug_*.jpg

---

### 6. debug_single_image.py
Debug complete pipeline on a single image with histogram visualization.

**Usage:**
```bash
cd debug
python debug_single_image.py
```

**Output:**
- Processes solidYellowCurve2 (configurable in script)
- Shows original, binary, warped binary, and histogram
- Displays lane base positions and distances
- Saves matplotlib visualization to ../output_images/

---

### 7. generate_docs_images.py
Generate all documentation images for README/writeup.

**Usage:**
```bash
cd debug
python generate_docs_images.py
```

**Output:**
- binary_combo_example.jpg (grayscale vs binary)
- warped_straight_lines.jpg (perspective transform)
- color_fit_lines.jpg (detected pixels + polynomials)
- example_output.jpg (final result)
- undistort_comparison.jpg (before/after calibration)
- Saves all to ../docs/

---

## Color Coding in Visualizations

- **RED channel** = Gradient (Sobel X)
- **GREEN channel** = Yellow lane pixels
- **BLUE channel** = White lane pixels

Where colors overlap:
- **CYAN** (blue + green) = Both white and yellow detected
- **YELLOW** (red + green) = Both gradient and yellow
- **MAGENTA** (red + blue) = Both gradient and white
- **WHITE** (all three) = All channels detected
