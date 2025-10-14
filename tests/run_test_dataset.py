#!/usr/bin/env python3
"""
Test Dataset Runner

This script automatically tests keypoint tracking across all subdirectories in /test_dataset.
For each subdirectory:
1. Find all images with corresponding JSON keypoint files
2. For each reference image (with JSON), track keypoints on all other images in the directory
3. Track both images with and without JSON files

The script uses the FlowFormer++ web API for tracking, similar to the example in
examples/ffpp_webapi_keypoint_tracker_example.py.

Usage:
    python tests/run_test_dataset.py
    python tests/run_test_dataset.py --url http://server:8001
    python tests/run_test_dataset.py --bidirectional
"""

import os
import sys
import cv2
import json
import time
import argparse
from pathlib import Path

# Add the parent directory to the path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.ffpp_webapi_keypoint_tracker import FFPPWebAPIKeypointTracker

# =============================================================================
# CONFIGURATION
# =============================================================================
WEB_SERVICE_URL = "http://msraig-ubuntu-2:8001"
TEST_DATASET_DIR = "test_dataset"


def load_keypoints_from_json(json_path):
    """
    Load keypoints from a JSON file.
    
    Args:
        json_path (str): Path to JSON file containing keypoints
        
    Returns:
        list: List of keypoint dictionaries with 'x' and 'y' coordinates
              Returns None if loading fails
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if 'keypoints' in data:
            keypoints = data['keypoints']
        elif isinstance(data, list):
            keypoints = data
        else:
            print(f"⚠️ Unexpected JSON format in {json_path}")
            return None
        
        # Ensure keypoints have required fields
        valid_keypoints = []
        for kp in keypoints:
            if 'x' in kp and 'y' in kp:
                valid_keypoints.append(kp)
            else:
                print(f"⚠️ Invalid keypoint format in {json_path}")
        
        return valid_keypoints if valid_keypoints else None
        
    except Exception as e:
        print(f"❌ Error loading keypoints from {json_path}: {e}")
        return None


def find_image_json_pairs(directory):
    """
    Find all images with corresponding JSON files in a directory.
    
    Args:
        directory (Path): Directory to search
        
    Returns:
        dict: Dictionary mapping image paths to their JSON keypoint files
              {image_path: json_path}
    """
    pairs = {}
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Find all images in directory
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in image_extensions:
            # Check if corresponding JSON exists
            json_path = file_path.with_suffix('.json')
            if json_path.exists():
                pairs[file_path] = json_path
    
    return pairs


def find_all_images(directory):
    """
    Find all image files in a directory.
    
    Args:
        directory (Path): Directory to search
        
    Returns:
        list: List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in image_extensions:
            images.append(file_path)
    
    return images


def test_subdirectory(subdir_path, tracker, bidirectional=False):
    """
    Test keypoint tracking for all images in a subdirectory.
    
    For each reference image with JSON:
        - Track keypoints on all other images (with and without JSON)
    
    Args:
        subdir_path (Path): Path to subdirectory
        tracker: FFPPWebAPIKeypointTracker instance
        bidirectional (bool): Whether to use bidirectional validation
        
    Returns:
        dict: Results dictionary containing tracking statistics and results
    """
    print(f"\n{'='*80}")
    print(f"📁 Processing subdirectory: {subdir_path.name}")
    print(f"{'='*80}")
    
    # Find images with JSON keypoints (reference images)
    ref_pairs = find_image_json_pairs(subdir_path)
    
    if not ref_pairs:
        print(f"⚠️ No image-JSON pairs found in {subdir_path.name}")
        return None
    
    # Find all images (including those without JSON)
    all_images = find_all_images(subdir_path)
    
    print(f"✅ Found {len(ref_pairs)} reference image(s) with keypoints")
    print(f"✅ Found {len(all_images)} total image(s) in directory")
    
    # Results storage
    results = {
        'subdirectory': subdir_path.name,
        'num_references': len(ref_pairs),
        'num_total_images': len(all_images),
        'tracking_results': []
    }
    
    # Process each reference image
    for ref_idx, (ref_image_path, ref_json_path) in enumerate(ref_pairs.items(), 1):
        print(f"\n{'─'*80}")
        print(f"🎯 Reference {ref_idx}/{len(ref_pairs)}: {ref_image_path.name}")
        print(f"{'─'*80}")
        
        # Load reference image
        ref_img = cv2.imread(str(ref_image_path))
        if ref_img is None:
            print(f"❌ Failed to load reference image: {ref_image_path}")
            continue
        
        # Load reference keypoints
        ref_keypoints = load_keypoints_from_json(ref_json_path)
        if ref_keypoints is None:
            print(f"❌ Failed to load keypoints from: {ref_json_path}")
            continue
        
        print(f"✅ Loaded reference: {ref_img.shape[1]}x{ref_img.shape[0]}")
        print(f"✅ Loaded {len(ref_keypoints)} keypoints")
        
        # Set reference in tracker
        ref_name = f"{subdir_path.name}_{ref_image_path.stem}"
        ref_start_time = time.time()
        ref_result = tracker.set_reference_image(ref_img, ref_keypoints, ref_name)
        ref_time = time.time() - ref_start_time
        
        if not ref_result.get('success'):
            print(f"❌ Failed to set reference: {ref_result.get('error')}")
            continue
        
        print(f"✅ Reference set in {ref_time:.3f}s")
        
        # Track on all other images
        target_images = [img for img in all_images if img != ref_image_path]
        
        print(f"\n📊 Tracking on {len(target_images)} target image(s)...")
        
        for target_idx, target_image_path in enumerate(target_images, 1):
            # Load target image
            target_img = cv2.imread(str(target_image_path))
            if target_img is None:
                print(f"❌ Failed to load target image: {target_image_path}")
                continue
            
            # Check if target has JSON (for comparison)
            target_json_path = target_image_path.with_suffix('.json')
            has_target_json = target_json_path.exists()
            
            # Track keypoints
            track_start_time = time.time()
            track_result = tracker.track_keypoints(
                target_img, 
                reference_name=ref_name,
                bidirectional=bidirectional
            )
            track_time = time.time() - track_start_time
            
            if track_result.get('success'):
                tracked_kps = len(track_result.get('tracked_keypoints', []))
                service_time = track_result.get('total_processing_time', 0)
                
                # Calculate statistics if bidirectional
                consistency_info = ""
                if bidirectional and 'bidirectional_stats' in track_result:
                    stats = track_result['bidirectional_stats']
                    mean_consistency = stats.get('mean_consistency_distance', 0)
                    high_acc = stats.get('high_accuracy_count', 0)
                    consistency_info = f" | Consistency: {mean_consistency:.2f}px ({high_acc} high-acc)"
                
                print(f"  [{target_idx}/{len(target_images)}] ✅ {target_image_path.name:<40} | "
                      f"Tracked: {tracked_kps:2d} | Time: {track_time:.3f}s | "
                      f"Has JSON: {'Yes' if has_target_json else 'No '}{consistency_info}")
                
                # Store result
                result_entry = {
                    'reference_image': ref_image_path.name,
                    'target_image': target_image_path.name,
                    'target_has_json': has_target_json,
                    'success': True,
                    'num_keypoints_tracked': tracked_kps,
                    'api_call_time': track_time,
                    'service_processing_time': service_time,
                    'bidirectional_enabled': bidirectional
                }
                
                if bidirectional and 'bidirectional_stats' in track_result:
                    result_entry['consistency_stats'] = track_result['bidirectional_stats']
                
                results['tracking_results'].append(result_entry)
                
            else:
                print(f"  [{target_idx}/{len(target_images)}] ❌ {target_image_path.name:<40} | "
                      f"Error: {track_result.get('error', 'Unknown')}")
                
                results['tracking_results'].append({
                    'reference_image': ref_image_path.name,
                    'target_image': target_image_path.name,
                    'target_has_json': has_target_json,
                    'success': False,
                    'error': track_result.get('error', 'Unknown')
                })
        
        # Clean up reference
        cleanup_result = tracker.remove_reference_image(ref_name)
        if cleanup_result.get('success'):
            print("✅ Reference cleaned up")
    
    return results


def run_test_dataset(service_url=WEB_SERVICE_URL, bidirectional=False):
    """
    Run tests on all subdirectories in test_dataset.
    
    Args:
        service_url (str): URL of FlowFormer++ web service
        bidirectional (bool): Whether to use bidirectional validation
        
    Returns:
        dict: Complete test results
    """
    print("🧪 Test Dataset Runner")
    print("=" * 80)
    print(f"Service URL: {service_url}")
    print(f"Bidirectional validation: {'Enabled' if bidirectional else 'Disabled'}")
    print(f"Dataset directory: {TEST_DATASET_DIR}")
    print("=" * 80)
    
    # Initialize tracker
    print("\n🚀 Initializing FFPPWebAPIKeypointTracker...")
    try:
        tracker = FFPPWebAPIKeypointTracker(service_url=service_url)
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return None
    
    # Check service health
    print("🏥 Checking service health...")
    health = tracker.get_service_health()
    if not health.get('success'):
        print(f"❌ Service not available: {health.get('error')}")
        return None
    
    print(f"✅ Service available: {health.get('status')}")
    
    # Find all subdirectories in test_dataset
    dataset_path = Path(TEST_DATASET_DIR)
    if not dataset_path.exists():
        print(f"❌ Test dataset directory not found: {TEST_DATASET_DIR}")
        return None
    
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"⚠️ No subdirectories found in {TEST_DATASET_DIR}")
        return None
    
    print(f"\n✅ Found {len(subdirs)} subdirectory(ies) to process")
    
    # Process each subdirectory
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'service_url': service_url,
        'bidirectional_enabled': bidirectional,
        'subdirectories': []
    }
    
    start_time = time.time()
    
    for subdir in sorted(subdirs):
        subdir_results = test_subdirectory(subdir, tracker, bidirectional)
        if subdir_results:
            all_results['subdirectories'].append(subdir_results)
    
    total_time = time.time() - start_time
    all_results['total_processing_time'] = total_time
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 Test Summary")
    print(f"{'='*80}")
    
    total_tracking_tests = 0
    successful_tests = 0
    
    for subdir_result in all_results['subdirectories']:
        subdir_name = subdir_result['subdirectory']
        num_results = len(subdir_result['tracking_results'])
        num_success = sum(1 for r in subdir_result['tracking_results'] if r.get('success'))
        
        total_tracking_tests += num_results
        successful_tests += num_success
        
        print(f"  {subdir_name}: {num_success}/{num_results} successful")
    
    print(f"\n  Total tracking tests: {total_tracking_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Failed: {total_tracking_tests - successful_tests}")
    print(f"  Success rate: {(successful_tests/total_tracking_tests*100):.1f}%")
    print(f"  Total time: {total_time:.2f}s")
    
    all_results['summary'] = {
        'total_tracking_tests': total_tracking_tests,
        'successful_tests': successful_tests,
        'failed_tests': total_tracking_tests - successful_tests,
        'success_rate': successful_tests / total_tracking_tests if total_tracking_tests > 0 else 0
    }
    
    return all_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run keypoint tracking tests on test_dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_test_dataset.py
  python tests/run_test_dataset.py --url http://server:8001
  python tests/run_test_dataset.py --bidirectional
  python tests/run_test_dataset.py --url http://gpu-server:8001 --bidirectional
        """
    )
    
    parser.add_argument('--url', '-u', type=str, 
                        help='FlowFormer++ web service URL (default: http://localhost:8001)')
    parser.add_argument('--bidirectional', '-b', action='store_true',
                        help='Enable bidirectional flow validation for accuracy assessment')
    
    args = parser.parse_args()
    
    # Use provided URL or default
    service_url = args.url if args.url else WEB_SERVICE_URL
    
    # Run tests
    results = run_test_dataset(service_url=service_url, bidirectional=args.bidirectional)
    
    if results is None:
        print("\n❌ Test run failed - check errors above")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("✅ Test dataset run completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
