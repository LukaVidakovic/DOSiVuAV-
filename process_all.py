#!/usr/bin/env python3
"""
Process all test images and videos through the lane detection pipeline
"""

import os
import sys
import cv2
import glob

sys.path.insert(0, 'src')

from pipeline import LaneDetectionPipeline

def process_all_images():
    """Process all test images"""
    print("=" * 60)
    print("Processing Test Images")
    print("=" * 60)
    
    pipeline = LaneDetectionPipeline("calibration.npz")
    
    # Get all test images
    image_paths = glob.glob("test_images/*.jpg")
    image_paths.sort()
    
    os.makedirs("output_images", exist_ok=True)
    
    results = []
    
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        print(f"\nProcessing: {basename}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ✗ Failed to load")
            continue
        
        result, stats = pipeline.process_image(img)
        
        if stats["detected"]:
            output_path = f"output_images/{basename}"
            cv2.imwrite(output_path, result)
            
            print(f"  ✓ Detected")
            print(f"    Curvature: {stats['avg_curvature']:.0f}m")
            print(f"    Offset: {stats['vehicle_offset']:.2f}m")
            print(f"    Left pixels: {stats['left_pixels']}")
            print(f"    Right pixels: {stats['right_pixels']}")
            
            results.append({
                'image': basename,
                'curvature': stats['avg_curvature'],
                'offset': stats['vehicle_offset'],
                'left_pixels': stats['left_pixels'],
                'right_pixels': stats['right_pixels']
            })
        else:
            print(f"  ✗ Detection failed")
    
    print(f"\n✓ Processed {len(results)}/{len(image_paths)} images successfully")
    return results


def process_all_videos():
    """Process all test videos"""
    print("\n" + "=" * 60)
    print("Processing Test Videos")
    print("=" * 60)
    
    pipeline = LaneDetectionPipeline("calibration.npz")
    
    # Get all test videos
    video_paths = glob.glob("test_videos/*.mp4")
    video_paths.sort()
    
    os.makedirs("output_videos", exist_ok=True)
    
    for video_path in video_paths:
        basename = os.path.basename(video_path)
        output_path = f"output_videos/{basename}"
        
        print(f"\nProcessing: {basename}")
        
        try:
            pipeline.process_video(video_path, output_path, display=False)
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    # Process images
    image_results = process_all_images()
    
    # Process videos
    if "--videos" in sys.argv:
        process_all_videos()
    else:
        print("\nSkipping videos (use --videos to process)")
    
    print("\n" + "=" * 60)
    print("✓ All processing complete!")
    print("=" * 60)
