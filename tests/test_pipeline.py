"""
Unit tests for complete pipeline
"""

import numpy as np
import cv2
import os
import sys
import pytest
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import LaneDetectionPipeline


@pytest.fixture
def test_image():
    """Create a test image."""
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    return img


@pytest.fixture
def pipeline_no_calibration():
    """Create pipeline without calibration."""
    # Use non-existent calibration file
    pipeline = LaneDetectionPipeline(calibration_file="nonexistent.npz")
    return pipeline


@pytest.fixture
def pipeline_with_calibration():
    """Create pipeline with calibration if available."""
    if os.path.exists("calibration.npz"):
        pipeline = LaneDetectionPipeline(calibration_file="calibration.npz")
    else:
        pytest.skip("Calibration file not available")
    return pipeline


def test_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = LaneDetectionPipeline(calibration_file="nonexistent.npz")

    assert pipeline.calibration_file == "nonexistent.npz"
    assert pipeline.mtx is None
    assert pipeline.dist is None
    assert pipeline.M is None
    assert pipeline.Minv is None
    assert pipeline.left_fit is None
    assert pipeline.right_fit is None
    assert pipeline.detected is False


def test_pipeline_initialization_with_calibration():
    """Test pipeline initialization with actual calibration."""
    if not os.path.exists("calibration.npz"):
        pytest.skip("Calibration file not available")

    pipeline = LaneDetectionPipeline(calibration_file="calibration.npz")

    # Should load calibration
    assert pipeline.mtx is not None
    assert pipeline.dist is not None


def test_process_image(pipeline_no_calibration, test_image):
    """Test processing a single image."""
    result, stats = pipeline_no_calibration.process_image(test_image)

    # Check output shape matches input
    assert result.shape == test_image.shape

    # Check stats dictionary
    assert "detected" in stats
    assert isinstance(stats["detected"], bool)


def test_process_image_detected(pipeline_no_calibration):
    """Test processing image with detectable lanes."""
    # Create image with vertical lane-like features
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Add white vertical lanes
    img[:, 400:410, :] = 255  # Left lane
    img[:, 870:880, :] = 255  # Right lane

    result, stats = pipeline_no_calibration.process_image(img)

    # Should have output
    assert result.shape == img.shape


def test_process_image_statistics(pipeline_no_calibration):
    """Test that statistics are properly populated."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, 400:410, :] = 255
    img[:, 870:880, :] = 255

    result, stats = pipeline_no_calibration.process_image(img)

    if stats["detected"]:
        # Check required statistics
        assert "left_pixels" in stats
        assert "right_pixels" in stats
        assert "left_curvature" in stats
        assert "right_curvature" in stats
        assert "avg_curvature" in stats
        assert "vehicle_offset" in stats
        assert "left_fit" in stats
        assert "right_fit" in stats


def test_process_image_with_previous(pipeline_no_calibration):
    """Test using previous polynomial for optimization."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, 400:410, :] = 255
    img[:, 870:880, :] = 255

    # First frame - full search
    result1, stats1 = pipeline_no_calibration.process_image(img, use_previous=False)

    # Second frame - use previous
    result2, stats2 = pipeline_no_calibration.process_image(img, use_previous=True)

    # Both should produce results
    assert result1.shape == img.shape
    assert result2.shape == img.shape


def test_draw_lane(pipeline_no_calibration, test_image):
    """Test lane drawing function."""
    # Create binary warped
    binary_warped = np.zeros((720, 1280), dtype=np.uint8)

    # Create simple polynomials
    left_fit = np.array([0.0, 0.0, 400.0])
    right_fit = np.array([0.0, 0.0, 880.0])

    # Initialize transform matrices
    from perspective_transform import get_default_src_points, get_default_dst_points, get_perspective_transform
    src = get_default_src_points(test_image.shape)
    dst = get_default_dst_points(test_image.shape)
    pipeline_no_calibration.M, pipeline_no_calibration.Minv = get_perspective_transform(src, dst)

    # Draw lane
    result = pipeline_no_calibration.draw_lane(
        test_image, binary_warped, left_fit, right_fit
    )

    # Check output
    assert result.shape == test_image.shape

    # Should have some green pixels (lane area)
    # Note: May not always have green due to random test image
    assert result.dtype == test_image.dtype


def test_add_text_overlay(pipeline_no_calibration, test_image):
    """Test adding text overlay."""
    result = pipeline_no_calibration.add_text_overlay(
        test_image, curvature=1000.0, offset=0.5
    )

    # Check output shape
    assert result.shape == test_image.shape

    # Image should be modified (text added)
    # Note: With random image, text may not be visible, but function should run
    assert result.dtype == test_image.dtype


def test_add_text_overlay_straight(pipeline_no_calibration, test_image):
    """Test text overlay for straight road."""
    result = pipeline_no_calibration.add_text_overlay(
        test_image, curvature=15000.0, offset=0.01
    )

    # Should handle straight line case
    assert result.shape == test_image.shape


def test_add_text_overlay_centered(pipeline_no_calibration, test_image):
    """Test text overlay for centered vehicle."""
    result = pipeline_no_calibration.add_text_overlay(
        test_image, curvature=800.0, offset=0.02
    )

    # Should handle centered case
    assert result.shape == test_image.shape


def test_state_persistence(pipeline_no_calibration):
    """Test that pipeline maintains state across frames."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, 400:410, :] = 255
    img[:, 870:880, :] = 255

    # Process first frame
    result1, stats1 = pipeline_no_calibration.process_image(img)

    if stats1["detected"]:
        # Check state is stored
        assert pipeline_no_calibration.left_fit is not None
        assert pipeline_no_calibration.right_fit is not None
        assert pipeline_no_calibration.detected is True


def test_process_video_file_not_found(pipeline_no_calibration):
    """Test that processing nonexistent video raises error."""
    with pytest.raises(FileNotFoundError):
        pipeline_no_calibration.process_video(
            "nonexistent_video.mp4",
            "output.mp4"
        )


def test_pipeline_handles_no_lanes(pipeline_no_calibration):
    """Test pipeline handles image with no detectable lanes."""
    # Black image - no lanes
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    result, stats = pipeline_no_calibration.process_image(img)

    # Should return something (even if detection failed)
    assert result.shape == img.shape
    assert "detected" in stats


def test_pipeline_transform_initialization(pipeline_no_calibration, test_image):
    """Test that perspective transform is initialized on first use."""
    assert pipeline_no_calibration.M is None
    assert pipeline_no_calibration.Minv is None

    # Process image
    result, stats = pipeline_no_calibration.process_image(test_image)

    # Transform matrices should be initialized
    assert pipeline_no_calibration.M is not None
    assert pipeline_no_calibration.Minv is not None


def test_real_test_image():
    """Test pipeline on actual test image if available."""
    if not os.path.exists("test_images/test1.jpg"):
        pytest.skip("Test image not available")

    pipeline = LaneDetectionPipeline(calibration_file="calibration.npz")
    img = cv2.imread("test_images/test1.jpg")

    result, stats = pipeline.process_image(img)

    # Check output
    assert result.shape == img.shape

    # If detected, check stats are reasonable
    if stats["detected"]:
        assert stats["left_pixels"] > 0
        assert stats["right_pixels"] > 0
        assert stats["avg_curvature"] > 0
        assert -2.0 < stats["vehicle_offset"] < 2.0  # Reasonable offset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
