#!/usr/bin/env python3
"""
3D Positioning Web API Example
===============================

Example script demonstrating how to use the 3D Positioning Service Web API
for multi-view triangulation with FlowFormer++ keypoint tracking.

Author: Yizhong Zhang
Date: November 2025
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
import threading

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.positioning_3d_webapi import Positioning3DWebAPIClient, load_camera_params_from_json


def test_upload_references():
    """
    Test function: Upload reference images to FFPP server.
    
    This function demonstrates how to trigger reference upload
    after the service has started.
    """
    print("=" * 60)
    print("Test: Upload Reference Images")
    print("=" * 60)
    
    # Initialize client
    client = Positioning3DWebAPIClient()
    
    # Check service health
    print("\n1. Checking service health...")
    health = client.check_health()
    
    if not health.get('success'):
        print(f"❌ Service not available: {health.get('error')}")
        return False
    
    print(f"✅ Service is running")
    print(f"   FFPP connected: {health.get('status', {}).get('ffpp_server', {}).get('connected')}")
    print(f"   References loaded: {health.get('status', {}).get('references', {}).get('loaded', 0)}")
    
    # List current references
    print("\n2. Listing current references...")
    refs = client.list_references()
    if refs.get('success'):
        ref_count = refs.get('count', 0)
        print(f"   Currently loaded: {ref_count} references")
        if ref_count > 0:
            for name in refs.get('references', {}).keys():
                print(f"   - {name}")
    
    # Upload references
    print("\n3. Uploading references to FFPP server...")
    result = client.upload_references()
    
    if result.get('success'):
        print(f"\n✅ Upload successful!")
        print(f"   References loaded: {result.get('references_loaded', 0)}")
        
        # List again to verify
        print("\n4. Verifying uploaded references...")
        refs = client.list_references()
        if refs.get('success'):
            print(f"   Total references: {refs.get('count', 0)}")
            for name, details in refs.get('references', {}).items():
                num_kp = details.get('metadata', {}).get('num_keypoints', 0)
                print(f"   - {name}: {num_kp} keypoints")
        
        return True
    else:
        print(f"\n❌ Upload failed: {result.get('error')}")
        return False


def test_triangulation_from_images(test_data_dir=None):
    """
    Test function: Upload images and perform triangulation.
    
    This function demonstrates the complete workflow:
    1. Initialize a session
    2. Upload multiple camera views with pose data
    3. Wait for tracking and triangulation
    4. Retrieve results
    
    Args:
        test_data_dir: Path to test data directory (default: dataset/ur_locate_push2end_data/test/test_img_20251118)
    """
    print("\n" + "=" * 60)
    print("Test: Triangulation from Images")
    print("=" * 60)
    
    # Configuration
    if test_data_dir is None:
        test_data_dir = Path("dataset/ur_locate_push2end_data/test/test_img_20251118")
    else:
        test_data_dir = Path(test_data_dir)
    
    # Extract reference_name from test_data_dir
    # Assuming structure: dataset/reference_name/test/...
    # We want to extract "reference_name" part
    parts = test_data_dir.parts
    if 'dataset' in parts:
        dataset_idx = parts.index('dataset')
        if dataset_idx + 1 < len(parts):
            reference_name = parts[dataset_idx + 1]
        else:
            print("❌ Cannot extract reference_name from test_data_dir")
            return False
    else:
        print("❌ test_data_dir should contain 'dataset' in path")
        return False
    
    # Check if test data exists
    if not test_data_dir.exists():
        print(f"❌ Test data directory not found: {test_data_dir}")
        return False
    
    # Find all images and their pose files
    image_files = sorted(test_data_dir.glob("*.jpg"))
    if not image_files:
        print(f"❌ No images found in {test_data_dir}")
        return False
    
    print(f"\n1. Found {len(image_files)} images in test directory")
    for img_file in image_files:
        print(f"   - {img_file.name}")
    
    # Initialize client
    client = Positioning3DWebAPIClient()
    
    # Check service health
    print("\n2. Checking service health...")
    health = client.check_health()
    if not health.get('success'):
        print(f"❌ Service not available: {health.get('error')}")
        return False
    
    print("✅ Service is running")
    
    # Check if reference exists
    print(f"\n3. Checking reference '{reference_name}'...")
    refs = client.list_references()
    if not refs.get('success') or reference_name not in refs.get('references', {}):
        print(f"❌ Reference '{reference_name}' not found")
        print(f"   Available references: {list(refs.get('references', {}).keys())}")
        return False
    
    print(f"✅ Reference '{reference_name}' is loaded")
    
    # Initialize session
    print(f"\n4. Initializing session...")
    session_result = client.init_session()
    
    if not session_result.get('success'):
        print(f"❌ Failed to initialize session: {session_result.get('error')}")
        return False
    
    session_id = session_result['session_id']
    print(f"✅ Session created: {session_id}")
    
    # Upload each view
    print(f"\n5. Uploading {len(image_files)} camera views...")
    for idx, img_file in enumerate(image_files):
        # Find corresponding pose file
        pose_file = img_file.parent / f"{img_file.stem}_pose.json"
        
        if not pose_file.exists():
            print(f"⚠️  Skipping {img_file.name}: pose file not found")
            continue
        
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"⚠️  Skipping {img_file.name}: failed to load image")
            continue
        
        # Load camera parameters
        try:
            intrinsic, distortion, extrinsic = load_camera_params_from_json(str(pose_file))
        except Exception as e:
            print(f"⚠️  Skipping {img_file.name}: failed to load pose - {e}")
            continue
        
        # Upload view
        print(f"   Uploading {img_file.name}...", end=" ")
        
        result = client.upload_view(
            session_id=session_id,
            reference_name=reference_name,
            image=image,
            intrinsic=intrinsic,
            distortion=distortion,
            extrinsic=extrinsic
        )
        
        if result.get('success'):
            print(f"✅ (queue position: {result.get('queue_position', 'N/A')})")
        else:
            print(f"❌ {result.get('error')}")
    
    # Wait for triangulation to complete and get results
    print("\n6. Waiting for tracking and triangulation (timeout: 30s)...")
    result = client.get_result(session_id, timeout=30000)  # 30 seconds
    
    success = False
    try:
        if not result.get('success'):
            print(f"❌ Failed to get result: {result.get('error')}")
            return False
        
        # Check if we got the final result or timed out
        if 'result' not in result:
            if result.get('timeout'):
                print("\n❌ Timeout waiting for triangulation")
            else:
                session_info = result.get('session', {})
                session_status = session_info.get('status')
                if session_status == 'failed':
                    print(f"\n❌ Session failed: {session_info.get('error_message', 'Unknown error')}")
                else:
                    print(f"\n❌ Triangulation not completed (status: {session_status})")
            return False
        
        print("✅ Triangulation completed!")
        
        triangulation_result = result['result']
        points_3d = np.array(triangulation_result['points_3d'])
        mean_error = triangulation_result['mean_error']
        processing_time = triangulation_result.get('processing_time', 0)
        
        print(f"\n✅ Triangulation Results:")
        print(f"   Number of 3D points: {len(points_3d)}")
        print(f"   Mean reprojection error: {mean_error:.3f} pixels")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"\n   Sample 3D points:")
        for i, pt in enumerate(points_3d[:5]):
            print(f"   Point {i}: ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
        
        if len(points_3d) > 5:
            print(f"   ... and {len(points_3d) - 5} more points")
        
        # Print per-view 2D keypoints with accuracy metrics (from result)
        print(f"\n   Number of views in result: {len(result.get('views', []))}")
        views_data = result.get('views', [])
        if views_data:
            print("\n   Per-view 2D keypoints:")
            for view_idx, view in enumerate(views_data):
                keypoints_2d = view.get('keypoints_2d', [])
                print(f"\n   View {view_idx}:")
                print(f"   Tracked {len(keypoints_2d)} keypoints:")
                
                # Show sample keypoints with consistency_distance
                for i, kp in enumerate(keypoints_2d[:3]):
                    x, y = kp.get('x'), kp.get('y')
                    consistency = kp.get('consistency_distance')
                    if consistency is not None:
                        print(f"      Point {i}: ({x:.2f}, {y:.2f}) - consistency_distance: {consistency:.3f}px")
                    else:
                        print(f"      Point {i}: ({x:.2f}, {y:.2f})")
                
                if len(keypoints_2d) > 3:
                    print(f"      ... and {len(keypoints_2d) - 3} more points")
                    
                    # Show consistency_distance statistics if available
                    consistencies = [kp.get('consistency_distance') for kp in keypoints_2d if kp.get('consistency_distance') is not None]
                    if consistencies:
                        print(f"      Consistency distance - mean: {np.mean(consistencies):.3f}px, min: {np.min(consistencies):.3f}px, max: {np.max(consistencies):.3f}px")
        
        success = True
        return True
        
    finally:
        # Clean up: terminate session (always execute, regardless of success/failure)
        print("\n7. Terminating session...")
        term_result = client.terminate_session(session_id)
        if term_result.get('success'):
            print("✅ Session terminated and cleaned up")
        else:
            print(f"⚠️  Failed to terminate session: {term_result.get('error')}")


def test_triangulation_incremental(test_data_dir=None):
    """
    Test function: Upload images incrementally and call get_result after each upload.
    
    This function demonstrates incremental triangulation workflow:
    1. Initialize a session
    2. Upload a view
    3. Call get_result (will fail with insufficient views initially)
    4. Sleep 3 seconds
    5. Repeat for all views except the last
    6. Upload last view and get final result
    
    Args:
        test_data_dir: Path to test data directory (default: dataset/ur_locate_push2end_data/test/test_img_20251118)
    """
    print("\n" + "=" * 60)
    print("Test: Incremental Triangulation (get_result after each upload)")
    print("=" * 60)
    
    # Configuration
    if test_data_dir is None:
        test_data_dir = Path("dataset/ur_locate_push2end_data/test/test_img_20251118")
    else:
        test_data_dir = Path(test_data_dir)
    
    # Extract reference_name from test_data_dir
    parts = test_data_dir.parts
    if 'dataset' in parts:
        dataset_idx = parts.index('dataset')
        if dataset_idx + 1 < len(parts):
            reference_name = parts[dataset_idx + 1]
        else:
            print("❌ Cannot extract reference_name from test_data_dir")
            return False
    else:
        print("❌ test_data_dir should contain 'dataset' in path")
        return False
    
    # Check if test data exists
    if not test_data_dir.exists():
        print(f"❌ Test data directory not found: {test_data_dir}")
        return False
    
    # Find all images and their pose files
    image_files = sorted(test_data_dir.glob("*.jpg"))
    if not image_files:
        print(f"❌ No images found in {test_data_dir}")
        return False
    
    print(f"\n1. Found {len(image_files)} images in test directory")
    for img_file in image_files:
        print(f"   - {img_file.name}")
    
    # Initialize client
    client = Positioning3DWebAPIClient()
    
    # Check service health
    print("\n2. Checking service health...")
    health = client.check_health()
    if not health.get('success'):
        print(f"❌ Service not available: {health.get('error')}")
        return False
    
    print("✅ Service is running")
    
    # Check if reference exists
    print(f"\n3. Checking reference '{reference_name}'...")
    refs = client.list_references()
    if not refs.get('success') or reference_name not in refs.get('references', {}):
        print(f"❌ Reference '{reference_name}' not found")
        print(f"   Available references: {list(refs.get('references', {}).keys())}")
        return False
    
    print(f"✅ Reference '{reference_name}' is loaded")
    
    # Initialize session
    print(f"\n4. Initializing session...")
    session_result = client.init_session()
    
    if not session_result.get('success'):
        print(f"❌ Failed to initialize session: {session_result.get('error')}")
        return False
    
    session_id = session_result['session_id']
    print(f"✅ Session created: {session_id}")
    
    success = False
    try:
        # Upload each view and call get_result (except for last view)
        print(f"\n5. Uploading {len(image_files)} camera views incrementally...")
        for view_idx, img_file in enumerate(image_files):
            is_last_view = (view_idx == len(image_files) - 1)
            
            # Find corresponding pose file
            pose_file = img_file.parent / f"{img_file.stem}_pose.json"
            
            if not pose_file.exists():
                print(f"⚠️  Skipping {img_file.name}: pose file not found")
                continue
            
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"⚠️  Skipping {img_file.name}: failed to load image")
                continue
            
            # Load camera parameters
            try:
                intrinsic, distortion, extrinsic = load_camera_params_from_json(str(pose_file))
            except Exception as e:
                print(f"⚠️  Skipping {img_file.name}: failed to load pose - {e}")
                continue
            
            # Upload view
            print(f"\n   [{view_idx+1}/{len(image_files)}] Uploading {img_file.name}...", end=" ")
            
            result = client.upload_view(
                session_id=session_id,
                reference_name=reference_name,
                image=image,
                intrinsic=intrinsic,
                distortion=distortion,
                extrinsic=extrinsic
            )
            
            if result.get('success'):
                print(f"✅ (queue position: {result.get('queue_position', 'N/A')})")
            else:
                print(f"❌ {result.get('error')}")
                continue
            
            # Call get_result after each upload
            if not is_last_view:
                print(f"   Calling get_result (timeout: 30s)...")
                result = client.get_result(session_id, timeout=30000)
                
                if result.get('success') and 'result' in result:
                    triangulation_result = result['result']
                    points_3d = np.array(triangulation_result['points_3d'])
                    mean_error = triangulation_result['mean_error']
                    num_views = len(result.get('views', []))
                    
                    # Show session info for debugging
                    session_info = result.get('session', {})
                    progress = session_info.get('progress', {})
                    views_received = progress.get('views_received', 0)
                    views_tracked = progress.get('views_tracked', 0)
                    
                    print(f"   ✅ Triangulated {len(points_3d)} points using {num_views} views (error: {mean_error:.3f}px)")
                    print(f"      Session has {views_received} received views, {views_tracked} tracked")
                else:
                    # Expected to fail for first view (insufficient views)
                    error_msg = result.get('error', 'Unknown error')
                    print(f"   ❌ {error_msg}")
                
                # Sleep 3 seconds before next upload
                print(f"   Sleeping 3 seconds...")
                time.sleep(3)
        
        # Get final result after all views uploaded
        print(f"\n6. Getting final triangulation result (timeout: 30s)...")
        result = client.get_result(session_id, timeout=30000)
        
        if not result.get('success'):
            print(f"❌ Failed to get result: {result.get('error')}")
            return False
        
        if 'result' not in result:
            print("\n❌ Triangulation not completed")
            return False
        
        print("✅ Final triangulation completed!")
        
        triangulation_result = result['result']
        points_3d = np.array(triangulation_result['points_3d'])
        mean_error = triangulation_result['mean_error']
        processing_time = triangulation_result.get('processing_time', 0)
        num_views = len(result.get('views', []))
        
        print(f"\n✅ Final Triangulation Results:")
        print(f"   Number of views used: {num_views}")
        print(f"   Number of 3D points: {len(points_3d)}")
        print(f"   Mean reprojection error: {mean_error:.3f} pixels")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"\n   Sample 3D points:")
        for i, pt in enumerate(points_3d[:5]):
            print(f"   Point {i}: ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
        
        if len(points_3d) > 5:
            print(f"   ... and {len(points_3d) - 5} more points")
        
        success = True
        return True
        
    finally:
        # Clean up: terminate session
        print("\n7. Terminating session...")
        term_result = client.terminate_session(session_id)
        if term_result.get('success'):
            print("✅ Session terminated and cleaned up")
        else:
            print(f"⚠️  Failed to terminate session: {term_result.get('error')}")


def test_triangulation_sequential():
    """
    Test function: Run triangulation sequentially on multiple test datasets.
    
    This function demonstrates running triangulation on different datasets
    in sequence to test the service's ability to handle multiple sessions.
    """
    print("\n" + "=" * 60)
    print("Test: Sequential Triangulation on Multiple Datasets")
    print("=" * 60)
    
    # Define test datasets
    test_datasets = [
        "dataset/ur_locate_push2end_data/test/test_img_20251118",
        "dataset/ur_locate_frame_data/test/session_1"
    ]
    
    results = []
    
    for idx, test_dir in enumerate(test_datasets, 1):
        print(f"\n{'=' * 60}")
        print(f"Running Test {idx}/{len(test_datasets)}: {test_dir}")
        print(f"{'=' * 60}")
        
        success = test_triangulation_from_images(test_data_dir=test_dir)
        results.append({
            'test_dir': test_dir,
            'success': success
        })
        
        if not success:
            print(f"\n⚠️  Test {idx} failed, continuing to next test...")
        else:
            print(f"\n✅ Test {idx} completed successfully!")
        
        # Brief pause between tests
        if idx < len(test_datasets):
            print("\nPausing 2 seconds before next test...")
            time.sleep(2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Sequential Triangulation Test Summary")
    print("=" * 60)
    
    successful_count = sum(1 for r in results if r['success'])
    
    for idx, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"Test {idx}: {status} - {result['test_dir']}")
    
    print(f"\nTotal: {successful_count}/{len(results)} tests passed")
    
    if successful_count == len(results):
        print("\n✅ All sequential tests passed!")
        return True
    else:
        print(f"\n⚠️  {len(results) - successful_count} test(s) failed")
        return False


def test_triangulation_concurrent():
    """
    Test function: Run triangulation concurrently on multiple test datasets.
    
    This function demonstrates running triangulation on different datasets
    simultaneously to test the service's ability to handle concurrent sessions.
    """
    print("\n" + "=" * 60)
    print("Test: Concurrent Triangulation on Multiple Datasets")
    print("=" * 60)
    
    # Define test datasets
    test_datasets = [
        "dataset/ur_locate_push2end_data/test/test_img_20251118",
        "dataset/ur_locate_frame_data/test/session_1"
    ]
    
    results = []
    threads = []
    results_lock = threading.Lock()
    
    def run_test(idx, test_dir):
        """Thread worker function to run a single test."""
        print(f"\n[Thread {idx}] Starting test for: {test_dir}")
        success = test_triangulation_from_images(test_data_dir=test_dir)
        
        with results_lock:
            results.append({
                'test_id': idx,
                'test_dir': test_dir,
                'success': success
            })
        
        if not success:
            print(f"\n[Thread {idx}] ⚠️  Test failed")
        else:
            print(f"\n[Thread {idx}] ✅ Test completed successfully!")
    
    # Start all threads
    print(f"\nStarting {len(test_datasets)} concurrent tests...")
    for idx, test_dir in enumerate(test_datasets, 1):
        thread = threading.Thread(
            target=run_test,
            args=(idx, test_dir),
            name=f"Test-{idx}"
        )
        threads.append(thread)
        thread.start()
        # Small stagger to avoid race conditions on startup
        time.sleep(0.1)
    
    # Wait for all threads to complete
    print("\nWaiting for all tests to complete...")
    for thread in threads:
        thread.join()
    
    # Summary
    print("\n" + "=" * 60)
    print("Concurrent Triangulation Test Summary")
    print("=" * 60)
    
    # Sort results by test_id for consistent display
    results.sort(key=lambda x: x['test_id'])
    
    successful_count = sum(1 for r in results if r['success'])
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"Test {result['test_id']}: {status} - {result['test_dir']}")
    
    print(f"\nTotal: {successful_count}/{len(results)} tests passed")
    
    if successful_count == len(results):
        print("\n✅ All concurrent tests passed!")
        return True
    else:
        print(f"\n⚠️  {len(results) - successful_count} test(s) failed")
        return False


def main():
    """Main function to run tests."""
    print("\n" + "=" * 60)
    print("3D Positioning Service - Web API Examples")
    print("=" * 60)
    
    # Test 1: Upload references
    print("\n[Test 1/5] Upload References")
    success1 = test_upload_references()
    
    # Test 2: Triangulation from images
    print("\n[Test 2/5] Triangulation from Images")
    success2 = test_triangulation_from_images()
    
    # Test 3: Incremental triangulation
    print("\n[Test 3/5] Incremental Triangulation")
    success3 = test_triangulation_incremental()
    
    # Test 4: Sequential triangulation
    print("\n[Test 4/5] Sequential Triangulation")
    success4 = test_triangulation_sequential()
    
    # Test 5: Concurrent triangulation
    print("\n[Test 5/5] Concurrent Triangulation")
    success5 = test_triangulation_concurrent()
    
    # Summary
    if success1 and success2 and success3 and success4 and success5:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Make sure the positioning service is running:")
        print("   python web/positioning_3d/app.py --ffpp-url http://msraig-ubuntu-3:8001 --dataset-path ./dataset")
        print("\n2. Check that FFPP server is accessible")
        print("\n3. Verify dataset path contains reference images:")
        print("   dataset/ref_name/ref_img_1.jpg")
        print("   dataset/ref_name/ref_img_1.json")
        print("\n4. Verify test images exist:")
        print("   dataset/ur_locate_push2end_data/test/test_img_20251118/*.jpg")
        print("   dataset/ur_locate_push2end_data/test/test_img_20251118/*_pose.json")


if __name__ == "__main__":
    main()
