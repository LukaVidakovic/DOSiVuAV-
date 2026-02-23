#!/usr/bin/env python3
"""Quick video test"""

import sys
sys.path.insert(0, '.')

from lane_detection_final import load_calibration, process_video

mtx, dist = load_calibration()

print("Processing project_video01.mp4...")
process_video('test_videos/project_video01.mp4', 
              'output_videos/project_video01.mp4', 
              mtx, dist)

print("\n✓ Done! Check output_videos/project_video01.mp4")
