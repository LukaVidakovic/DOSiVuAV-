"""
Complete Lane Detection Pipeline

This module integrates all components into a complete lane detection pipeline
that can process images and videos.
"""

import numpy as np
import cv2
import os
import argparse
from typing import Tuple

# Import all modules
from camera_calibration import load_calibration, undistort_image
from thresholding import combined_threshold
from perspective_transform import (
    get_default_src_points,
    get_default_dst_points,
    get_perspective_transform,
    warp_image,
    unwarp_image
)
from lane_detection import (
    find_lane_pixels,
    fit_polynomial,
    search_around_poly,
    calculate_curvature,
    calculate_vehicle_position
)


class LaneDetectionPipeline:
    """Complete lane detection pipeline with state management for video."""

    def __init__(self, calibration_file: str = "calibration.npz"):
        """
        Initialize pipeline.

        Args:
            calibration_file: Path to camera calibration file
        """
        self.calibration_file = calibration_file
        self.mtx = None
        self.dist = None
        self.M = None
        self.Minv = None

        # State for video processing (previous frame detection)
        self.left_fit = None
        self.right_fit = None
        self.detected = False

        # Load calibration if available
        if os.path.exists(calibration_file):
            self.mtx, self.dist, _, _ = load_calibration(calibration_file)
            print(f"✓ Loaded camera calibration from {calibration_file}")
        else:
            print(f"⚠ Camera calibration not found: {calibration_file}")

    def process_image(
        self,
        img: np.ndarray,
        use_previous: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Process a single image through the complete pipeline.

        Args:
            img: Input image (BGR)
            use_previous: Whether to use previous polynomial for search (video optimization)

        Returns:
            result: Output image with lane overlay
            stats: Dictionary with detection statistics
        """
        # 1. Undistort
        if self.mtx is not None:
            undist = undistort_image(img, self.mtx, self.dist)
        else:
            undist = img.copy()

        # 2. Binary thresholding
        binary = combined_threshold(undist)

        # 3. Perspective transform
        if self.M is None:
            src = get_default_src_points(img.shape)
            dst = get_default_dst_points(img.shape)
            self.M, self.Minv = get_perspective_transform(src, dst)

        binary_warped = warp_image(binary, self.M)

        # 4. Find lane pixels
        if use_previous and self.detected and self.left_fit is not None:
            # Use previous polynomial for faster search
            leftx, lefty, rightx, righty = search_around_poly(
                binary_warped, self.left_fit, self.right_fit
            )
        else:
            # Full sliding window search
            leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

        # 5. Fit polynomial
        if len(leftx) > 0 and len(rightx) > 0:
            left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
            self.left_fit = left_fit
            self.right_fit = right_fit
            self.detected = True
        else:
            # No lanes detected, use previous if available
            if self.left_fit is None:
                # No previous fit, can't proceed
                return img.copy(), {"detected": False}
            left_fit = self.left_fit
            right_fit = self.right_fit

        # 6. Calculate curvature and vehicle position
        left_curv, right_curv = calculate_curvature(
            left_fit, right_fit, img.shape[0]-1
        )
        avg_curvature = (left_curv + right_curv) / 2
        vehicle_offset = calculate_vehicle_position(left_fit, right_fit, img.shape[1])

        # 7. Draw lane on image
        result = self.draw_lane(undist, binary_warped, left_fit, right_fit)

        # 8. Add text overlay
        result = self.add_text_overlay(
            result, avg_curvature, vehicle_offset
        )

        # Compile statistics
        stats = {
            "detected": True,
            "left_pixels": len(leftx),
            "right_pixels": len(rightx),
            "left_curvature": left_curv,
            "right_curvature": right_curv,
            "avg_curvature": avg_curvature,
            "vehicle_offset": vehicle_offset,
            "left_fit": left_fit,
            "right_fit": right_fit
        }

        return result, stats

    def draw_lane(
        self,
        undist: np.ndarray,
        binary_warped: np.ndarray,
        left_fit: np.ndarray,
        right_fit: np.ndarray
    ) -> np.ndarray:
        """
        Draw detected lane area on the original image.

        Args:
            undist: Undistorted original image
            binary_warped: Binary warped image (for reference)
            left_fit: Left lane polynomial
            right_fit: Right lane polynomial

        Returns:
            Image with lane overlay
        """
        # Create image to draw lanes
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Generate y values
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

        # Calculate x values from polynomials
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Recast x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw lane onto warped blank image
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

        # Draw lane lines
        cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), thickness=25)
        cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), thickness=25)

        # Warp back to original image space
        newwarp = unwarp_image(color_warp, self.Minv, (undist.shape[1], undist.shape[0]))

        # Combine with original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result

    def add_text_overlay(
        self,
        img: np.ndarray,
        curvature: float,
        offset: float
    ) -> np.ndarray:
        """
        Add text overlay with curvature and vehicle position.

        Args:
            img: Input image
            curvature: Lane curvature radius (meters)
            offset: Vehicle offset from center (meters)

        Returns:
            Image with text overlay
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        color = (255, 255, 255)

        # Curvature text
        if curvature > 10000:
            curv_text = "Radius of Curvature: Straight"
        else:
            curv_text = f"Radius of Curvature: {curvature:.0f}m"

        cv2.putText(img, curv_text, (50, 50), font, font_scale, color, font_thickness)

        # Vehicle position text
        if abs(offset) < 0.05:
            pos_text = "Vehicle Position: Centered"
        elif offset < 0:
            pos_text = f"Vehicle Position: {abs(offset):.2f}m left of center"
        else:
            pos_text = f"Vehicle Position: {offset:.2f}m right of center"

        cv2.putText(img, pos_text, (50, 100), font, font_scale, color, font_thickness)

        return img

    def process_video(
        self,
        input_path: str,
        output_path: str,
        display: bool = False
    ) -> None:
        """
        Process video file through pipeline.

        Args:
            input_path: Path to input video
            output_path: Path to save output video
            display: Whether to display frames during processing
        """
        # Open video
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {input_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset state for new video
        self.left_fit = None
        self.right_fit = None
        self.detected = False

        frame_count = 0
        success_count = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Process frame (use previous polynomial for speed)
                result, stats = self.process_image(frame, use_previous=True)

                if stats["detected"]:
                    success_count += 1

                # Write frame
                out.write(result)

                # Display progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

                # Display frame if requested
                if display:
                    cv2.imshow('Lane Detection', result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()

        print("\n✓ Video processing complete!")
        print(f"  Output: {output_path}")
        print(f"  Frames processed: {frame_count}")
        print(f"  Successful detections: {success_count} ({success_count/frame_count*100:.1f}%)")


def main():
    """Command-line interface for lane detection pipeline."""
    parser = argparse.ArgumentParser(description="Advanced Lane Detection Pipeline")

    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to input image"
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to input video"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to output file (auto-generated if not specified)"
    )

    parser.add_argument(
        "--calibration", "-c",
        type=str,
        default="calibration.npz",
        help="Path to camera calibration file (default: calibration.npz)"
    )

    parser.add_argument(
        "--display", "-d",
        action="store_true",
        help="Display output during video processing"
    )

    args = parser.parse_args()

    # Validate input
    if not args.image and not args.video:
        parser.error("Must specify either --image or --video")

    # Initialize pipeline
    pipeline = LaneDetectionPipeline(calibration_file=args.calibration)

    # Process image
    if args.image:
        if not os.path.exists(args.image):
            print(f"✗ Error: Image not found: {args.image}")
            return

        # Load image
        img = cv2.imread(args.image)

        # Process
        result, stats = pipeline.process_image(img)

        # Generate output path
        if args.output:
            output_path = args.output
        else:
            base, ext = os.path.splitext(args.image)
            output_path = f"{base}_output{ext}"

        # Save
        cv2.imwrite(output_path, result)

        print("\n✓ Image processing complete!")
        print(f"  Output: {output_path}")
        if stats["detected"]:
            print(f"  Curvature: {stats['avg_curvature']:.0f}m")
            print(f"  Vehicle offset: {stats['vehicle_offset']:.2f}m")

    # Process video
    elif args.video:
        if not os.path.exists(args.video):
            print(f"✗ Error: Video not found: {args.video}")
            return

        # Generate output path
        if args.output:
            output_path = args.output
        else:
            base, ext = os.path.splitext(args.video)
            output_path = f"{base}_output.mp4"

        # Process
        pipeline.process_video(args.video, output_path, display=args.display)


if __name__ == "__main__":
    main()
