#!/usr/bin/env python3
"""
3D Coordinate Estimation Example
===============================

A comprehensive script that demonstrates the complete computer vision pipeline:
1. Keypoint tracking using FFPPKeypointTracker on multiple images
2. 3D coordinate estimation via triangulation from multiple camera views

This script automatically processes all numbered image files in the 
temp/positioning_data directory (e.g., 1.jpg, 2.jpg, 10.jpg, etc.)
and their corresponding pose files, then estimates 3D coordinates
of tracked keypoints using camera calibration parameters.

Features:
- Dynamic detection of input images (no hardcoded limits)
- Automatic keypoint tracking across multiple views
- 3D triangulation from tracked 2D keypoints
- Comprehensive error handling and progress reporting

Usage:
    python examples/estimate_3d_coordinates_example.py
"""

import os
import sys
import json
import re
import glob
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from core.ffpp_keypoint_tracker import FFPPKeypointTracker

# Import 3D coordinate estimation function  
sys.path.append(str(project_root / "core"))
import importlib.util
spec = importlib.util.spec_from_file_location("coordinates_estimation", project_root / "core" / "3d_coordinates_estimation.py")
coordinates_estimation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinates_estimation)
estimate_3d = coordinates_estimation.estimate_3d


def load_image(image_path):
    """Load image and convert to RGB numpy array."""
    with Image.open(image_path) as img:
        return np.array(img.convert('RGB'))


def load_keypoints_from_json(json_path):
    """Load keypoints from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    keypoints = []
    for kp in data.get('keypoints', []):
        keypoints.append({
            'x': kp['x'],
            'y': kp['y'],
            'id': kp.get('id', len(keypoints) + 1),
            'name': kp.get('name', f'point_{len(keypoints) + 1}')
        })
    
    return keypoints


def run_keypoint_tracking():
    """Main function to run keypoint tracking examples."""
    print("🚀 Starting Keypoint Tracking Examples")
    print("=" * 50)
    
    # Setup paths
    data_dir = project_root / "temp" / "positioning_data"
    ref_image_path = data_dir / "ref_img.jpg"
    ref_keypoints_path = data_dir / "ref_keypoints.json"
    
    # Check if reference files exist
    if not ref_image_path.exists():
        print(f"❌ Reference image not found: {ref_image_path}")
        return
    
    if not ref_keypoints_path.exists():
        print(f"❌ Reference keypoints not found: {ref_keypoints_path}")
        return
    
    # Initialize tracker
    print("🔧 Initializing FFPPKeypointTracker...")
    tracker = FFPPKeypointTracker()
    
    # Load reference image and keypoints
    print("📖 Loading reference image and keypoints...")
    ref_image = load_image(ref_image_path)
    ref_keypoints = load_keypoints_from_json(ref_keypoints_path)
    
    print(f"   Reference image shape: {ref_image.shape}")
    print(f"   Number of keypoints: {len(ref_keypoints)}")
    
    # Set reference image
    print("🎯 Setting reference image...")
    ref_result = tracker.set_reference_image(ref_image, ref_keypoints, "reference")
    
    if not ref_result['success']:
        print(f"❌ Failed to set reference image: {ref_result.get('error', 'Unknown error')}")
        return
    
    print("✅ Reference image set successfully")
    
    # Find all target images dynamically (any number.jpg format)
    target_files = []
    
    # Get all .jpg files that match numeric pattern (e.g., 1.jpg, 2.jpg, 10.jpg, etc.)
    jpg_pattern = str(data_dir / "*.jpg")
    jpg_files = glob.glob(jpg_pattern)
    
    for jpg_file in jpg_files:
        jpg_path = Path(jpg_file)
        filename = jpg_path.stem  # Get filename without extension
        
        # Check if filename is a number (skip ref_img.jpg)
        if filename.isdigit():
            json_path = data_dir / f"{filename}.json"
            if json_path.exists():
                target_files.append((jpg_path, json_path))
    
    # Sort by numeric value for consistent processing order
    target_files.sort(key=lambda x: int(x[0].stem))
    
    if not target_files:
        print("❌ No target image files found (expected format: number.jpg with corresponding number.json)")
        print(f"   Searched in: {data_dir}")
        return
    
    print(f"📷 Found {len(target_files)} target images")
    print()
    
    # Track keypoints for each target image
    successful_tracks = 0
    output_dir = data_dir  # Save results in the same directory as input data
    # No need to create directory as it already exists
    
    for img_path, json_path in target_files:
        print(f"🔍 Tracking keypoints for {img_path.name}...")
        
        # Load target image
        target_image = load_image(img_path)
        
        # Load corresponding pose data (for context)
        with open(json_path, 'r') as f:
            pose_data = json.load(f)
        
        # Track keypoints
        track_result = tracker.track_keypoints(target_image)
        
        if track_result['success']:
            tracked_kps = track_result['tracked_keypoints']
            print(f"   ✅ Successfully tracked {len(tracked_kps)} keypoints")
            print(f"   ⏱️  Processing time: {track_result.get('processing_time', 0):.3f}s")
            
            # Prepare result for this image
            result = {
                'image_file': str(img_path.name),
                'tracking_result': track_result
            }
            
            # Save individual result file
            output_filename = f"{img_path.stem}_tracking_result.json"
            output_file = output_dir / output_filename
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"   💾 Result saved to: {output_file}")
            successful_tracks += 1
        else:
            print(f"   ❌ Tracking failed: {track_result.get('error', 'Unknown error')}")
        
        print()
    
    # Summary
    print("📊 TRACKING SUMMARY")
    print("=" * 30)
    print(f"Total target images: {len(target_files)}")
    print(f"Successful tracks: {successful_tracks}")
    print(f"Output directory: {output_dir}")
    
    return successful_tracks


def run_3d_coordinate_estimation(successful_tracks):
    """Run 3D coordinate estimation after keypoint tracking."""
    if successful_tracks < 2:
        print("⚠️  Skipping 3D coordinate estimation: Need at least 2 successful tracking results")
        return False
    
    print("\n" + "=" * 50)
    print("🧮 Starting 3D Coordinate Estimation")
    print("=" * 50)
    
    # Setup paths
    data_dir = project_root / "temp" / "positioning_data"
    camera_params_dir = project_root / "temp" / "camera_parameters"
    output_dir = project_root / "temp" / "3d_coordinate_estimation_result"
    
    # Reference files
    ref_keypoints_file = str(data_dir / "ref_keypoints.json")
    ref_pose_file = str(data_dir / "ref_pose.json")
    
    # Camera parameter files
    intrinsic_file = str(camera_params_dir / "calibration_result.json")
    extrinsic_file = str(camera_params_dir / "eye_in_hand_result.json")
    
    # Check if reference files exist
    if not Path(ref_keypoints_file).exists():
        print(f"❌ Reference keypoints file not found: {ref_keypoints_file}")
        return False
    
    if not Path(ref_pose_file).exists():
        print(f"❌ Reference pose file not found: {ref_pose_file}")
        return False
    
    if not Path(intrinsic_file).exists():
        print(f"❌ Camera intrinsic file not found: {intrinsic_file}")
        return False
    
    if not Path(extrinsic_file).exists():
        print(f"❌ Camera extrinsic file not found: {extrinsic_file}")
        return False
    
    # Find all successful tracking result files dynamically
    view_configs = []
    
    # Get all tracking result files (pattern: number_tracking_result.json)
    tracking_pattern = str(data_dir / "*_tracking_result.json")
    tracking_files = glob.glob(tracking_pattern)
    
    for tracking_file in tracking_files:
        tracking_path = Path(tracking_file)
        # Extract the number from filename (e.g., "1_tracking_result.json" -> "1")
        filename = tracking_path.name
        if filename.endswith('_tracking_result.json'):
            number_part = filename.replace('_tracking_result.json', '')
            
            # Check if it's a valid number
            if number_part.isdigit():
                pose_file = data_dir / f"{number_part}.json"
                
                if pose_file.exists():
                    view_config = {
                        'keypoint_file': str(tracking_path),
                        'pose_file': str(pose_file),
                        'intrinsic_file': intrinsic_file,
                        'extrinsic_file': extrinsic_file
                    }
                    view_configs.append((int(number_part), view_config))
    
    # Sort by number for consistent processing order
    view_configs.sort(key=lambda x: x[0])
    view_configs = [config for _, config in view_configs]  # Remove the sorting key
    
    if len(view_configs) < 2:
        print(f"❌ Need at least 2 view configurations, but only found {len(view_configs)}")
        return False
    
    print(f"📋 Found {len(view_configs)} valid view configurations")
    print(f"📁 Reference keypoints: {ref_keypoints_file}")
    print(f"📁 Reference pose: {ref_pose_file}")
    print(f"📁 Camera intrinsic: {intrinsic_file}")
    print(f"📁 Camera extrinsic: {extrinsic_file}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Run 3D coordinate estimation
    try:
        print("🚀 Running 3D coordinate estimation...")
        results = estimate_3d(
            view_configs=view_configs,
            reference_keypoints_file=ref_keypoints_file,
            reference_pose_file=ref_pose_file,
            output_file=str(output_dir),
            model_path=None,
            device='auto'
        )
        
        if results['success']:
            print("✅ 3D coordinate estimation completed successfully!")
            print(f"⏱️  Processing time: {results.get('processing_time', 0):.3f}s")
            
            # Display results summary
            if 'keypoints_3d' in results:
                keypoints_3d = results['keypoints_3d']
                print(f"📊 Estimated 3D coordinates for {len(keypoints_3d)} keypoints:")
                print("-" * 50)
                
                for kp in keypoints_3d:
                    coords = kp['coordinates_3d']
                    print(f"  {kp['name']} (ID: {kp['id']}):")
                    print(f"    X: {coords['x']:.6f} m")
                    print(f"    Y: {coords['y']:.6f} m") 
                    print(f"    Z: {coords['z']:.6f} m")
                    print()
            
            return True
        else:
            print(f"❌ 3D coordinate estimation failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during 3D coordinate estimation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    try:
        # Step 1: Run keypoint tracking
        successful_tracks = run_keypoint_tracking()
        
        # Step 2: Run 3D coordinate estimation if tracking was successful
        if successful_tracks > 0:
            print("\n" + "=" * 60)
            print("🔄 Proceeding to 3D coordinate estimation...")
            print("=" * 60)
            
            estimation_success = run_3d_coordinate_estimation(successful_tracks)
            
            if estimation_success:
                print("\n🎉 Complete pipeline finished successfully!")
                print("✅ Keypoint tracking: COMPLETED")
                print("✅ 3D coordinate estimation: COMPLETED")
            else:
                print("\n⚠️  Pipeline partially completed:")
                print("✅ Keypoint tracking: COMPLETED")
                print("❌ 3D coordinate estimation: FAILED")
        else:
            print("\n⚠️  No successful keypoint tracking results found.")
            print("❌ Cannot proceed with 3D coordinate estimation.")
        
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())