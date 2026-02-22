# Setup Guide

## Installation

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or on Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## Running Camera Calibration

```bash
# Activate virtual environment first
source venv/bin/activate

# Run calibration
python3 src/camera_calibration.py
```

This will:
- Process all images in `camera_cal/`
- Create `calibration.npz` with camera parameters
- Generate `docs/undistort_comparison.jpg` showing before/after

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_camera_calibration.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── camera_calibration.py
│   └── (more modules...)
├── tests/                  # Unit tests
│   ├── test_camera_calibration.py
│   └── (more tests...)
├── camera_cal/            # Calibration images
├── test_images/           # Test images
├── test_videos/           # Test videos
├── docs/                  # Documentation images
├── output_images/         # Processed images
├── output_videos/         # Processed videos
├── requirements.txt       # Dependencies
└── README.md             # Project documentation
```
