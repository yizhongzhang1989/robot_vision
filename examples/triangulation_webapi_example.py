#!/usr/bin/env python3
"""
3D Positioning Web API Example
===============================

Example script demonstrating how to use the 3D Positioning Service Web API
for multi-view triangulation with FlowFormer++ keypoint tracking.

This script provides a wrapper class for easy interaction with the service
and includes example usage for testing triangulation.

Author: Yizhong Zhang
Date: November 2025
"""

import requests
import json
import time
import base64
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from pathlib import Path


class PositioningServiceClient:
    """
    Client wrapper for the 3D Positioning Service Web API.
    
    Provides convenient methods for:
    - Uploading reference images
    - Creating positioning sessions
    - Uploading camera views
    - Retrieving triangulation results
    """
    
    def __init__(self, service_url: str = "http://localhost:8004"):
        """
        Initialize the client.
        
        Args:
            service_url: Base URL of the positioning service
        """
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
    
    def check_health(self) -> Dict:
        """
        Check if the service is running and healthy.
        
        Returns:
            Health status dictionary
        """
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                'success': False,
                'error': f'Service not reachable: {str(e)}'
            }
    
    def upload_references(self) -> Dict:
        """
        Trigger manual upload of reference images to FFPP server.
        
        This will parse the dataset directory and upload all reference images
        to the FFPP server for keypoint tracking.
        
        Returns:
            Upload result with success status and count
        """
        try:
            response = self.session.post(
                f"{self.service_url}/upload_references",
                headers={'Content-Type': 'application/json'},
                timeout=120  # Upload can take time
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                print(f"✅ Successfully uploaded {result.get('references_loaded', 0)} references")
                print(f"   Total found: {result.get('references_found', 0)}")
            else:
                print(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Upload timeout - check if FFPP server is responding'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Upload error: {str(e)}'
            }
    
    def list_references(self) -> Dict:
        """
        List all available reference images.
        
        Returns:
            Dictionary with reference names and details
        """
        try:
            response = self.session.get(f"{self.service_url}/list_references")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_queue_status(self) -> Dict:
        """
        Get current queue status.
        
        Returns:
            Queue statistics
        """
        try:
            response = self.session.get(f"{self.service_url}/queue_status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def init_session(self, robot_id: str, reference_name: str, num_expected_views: int) -> Dict:
        """
        Initialize a new positioning session.
        
        Args:
            robot_id: Unique identifier for the robot
            reference_name: Name of the reference image to use
            num_expected_views: Number of camera views expected
            
        Returns:
            Response with session_id if successful
        """
        try:
            payload = {
                'robot_id': robot_id,
                'reference_name': reference_name,
                'num_expected_views': num_expected_views
            }
            
            response = self.session.post(
                f"{self.service_url}/init_session",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def upload_view(self, session_id: str, view_id: str, image: np.ndarray,
                   intrinsic: np.ndarray, extrinsic: np.ndarray,
                   distortion: Optional[np.ndarray] = None,
                   image_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Upload a camera view for triangulation.
        
        Args:
            session_id: Session identifier from init_session
            view_id: Unique identifier for this view
            image: Camera image as numpy array (BGR or RGB)
            intrinsic: 3x3 camera intrinsic matrix
            extrinsic: 4x4 camera extrinsic matrix (camera to world)
            distortion: Distortion coefficients (optional)
            image_size: Image size as (width, height) (optional, inferred from image)
            
        Returns:
            Response with queue position if successful
        """
        try:
            # Convert image to base64
            image_base64 = image_to_base64(image)
            
            # Infer image size if not provided
            if image_size is None:
                h, w = image.shape[:2]
                image_size = (w, h)
            
            # Prepare camera parameters
            camera_params = {
                'intrinsic': intrinsic.tolist(),
                'extrinsic': extrinsic.tolist(),
                'image_size': list(image_size)
            }
            
            if distortion is not None:
                camera_params['distortion'] = distortion.tolist()
            
            # Prepare payload
            payload = {
                'session_id': session_id,
                'view_id': view_id,
                'image_base64': image_base64,
                'camera_params': camera_params
            }
            
            response = self.session.post(
                f"{self.service_url}/upload_view",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_session_status(self, session_id: str) -> Dict:
        """
        Get status of a positioning session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session status with progress information
        """
        try:
            response = self.session.get(f"{self.service_url}/session_status/{session_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_result(self, session_id: str) -> Dict:
        """
        Get triangulation result for a completed session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Triangulation result with 3D points
        """
        try:
            response = self.session.get(f"{self.service_url}/result/{session_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert numpy image to base64 encoded string.
    
    Args:
        image: Image as numpy array (BGR or RGB)
        
    Returns:
        Base64 encoded image string with data URI prefix
    """
    # Ensure RGB format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume BGR from OpenCV, convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Encode as PNG
    success, buffer = cv2.imencode('.png', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Failed to encode image")
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


def load_camera_params_from_json(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load camera parameters from pose JSON file.
    
    Args:
        json_path: Path to the pose JSON file
        
    Returns:
        Tuple of (intrinsic_matrix, distortion_coeffs, extrinsic_matrix)
        where extrinsic = end2base @ cam2end_matrix
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract camera matrix (intrinsic)
    intrinsic = np.array(data['camera_matrix'], dtype=np.float64)
    
    # Extract distortion coefficients
    distortion = np.array(data['distortion_coefficients'], dtype=np.float64)
    
    # Extract transformation matrices
    end2base = np.array(data['end2base'], dtype=np.float64)
    cam2end = np.array(data['cam2end_matrix'], dtype=np.float64)
    
    # Calculate extrinsic: cam2base = end2base @ cam2end
    extrinsic = end2base @ cam2end
    
    return intrinsic, distortion, extrinsic


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
    client = PositioningServiceClient()
    
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


def test_triangulation_from_images():
    """
    Test function: Upload images and perform triangulation.
    
    This function demonstrates the complete workflow:
    1. Initialize a session
    2. Upload multiple camera views with pose data
    3. Wait for tracking and triangulation
    4. Retrieve results
    """
    print("\n" + "=" * 60)
    print("Test: Triangulation from Images")
    print("=" * 60)
    
    # Configuration
    test_data_dir = Path("output/dataset/ur_locate_push2end_data/test/test_img_20251118")
    reference_name = "ur_locate_push2end_data"
    robot_id = "test_robot_01"
    
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
    client = PositioningServiceClient()
    
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
    print(f"\n4. Initializing session for {len(image_files)} views...")
    session_result = client.init_session(
        robot_id=robot_id,
        reference_name=reference_name,
        num_expected_views=len(image_files)
    )
    
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
        view_id = f"view_{idx}"
        print(f"   Uploading {img_file.name} as {view_id}...", end=" ")
        
        result = client.upload_view(
            session_id=session_id,
            view_id=view_id,
            image=image,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            distortion=distortion
        )
        
        if result.get('success'):
            print(f"✅ (queue position: {result.get('queue_position', 'N/A')})")
        else:
            print(f"❌ {result.get('error')}")
    
    # Wait for triangulation to complete
    print("\n6. Waiting for tracking and triangulation...")
    max_wait_time = 120  # seconds
    check_interval = 2  # seconds
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        status = client.get_session_status(session_id)
        
        if not status.get('success'):
            print(f"❌ Failed to get status: {status.get('error')}")
            return False
        
        session_info = status['session']
        session_status = session_info['status']
        progress = session_info['progress']
        
        print(f"   Status: {session_status} | Progress: {progress['views_tracked']}/{progress['expected_views']} views tracked", end="\r")
        
        if session_status == 'completed':
            print("\n✅ Triangulation completed!")
            break
        elif session_status == 'failed':
            print(f"\n❌ Session failed: {session_info.get('error_message', 'Unknown error')}")
            return False
        
        time.sleep(check_interval)
        elapsed_time += check_interval
    
    if elapsed_time >= max_wait_time:
        print("\n❌ Timeout waiting for triangulation")
        return False
    
    # Get results
    print("\n7. Retrieving triangulation results...")
    result = client.get_result(session_id)
    
    if not result.get('success'):
        print(f"❌ Failed to get result: {result.get('error')}")
        return False
    
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
    
    return True


def main():
    """Main function to run tests."""
    print("\n" + "=" * 60)
    print("3D Positioning Service - Web API Examples")
    print("=" * 60)
    
    # Test 1: Upload references
    print("\n[Test 1/2] Upload References")
    success1 = test_upload_references()
    
    # Test 2: Triangulation from images
    print("\n[Test 2/2] Triangulation from Images")
    success2 = test_triangulation_from_images()
    
    # Summary
    if success1 and success2:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Make sure the positioning service is running:")
        print("   python web/positioning_3d/app.py --ffpp-url http://msraig-ubuntu-3:8001")
        print("\n2. Check that FFPP server is accessible")
        print("\n3. Verify dataset path contains reference images:")
        print("   output/dataset/ref_name/ref_img_1.jpg")
        print("   output/dataset/ref_name/ref_img_1.json")
        print("\n4. Verify test images exist:")
        print("   output/dataset/ur_locate_push2end_data/test/test_img_20251118/*.jpg")
        print("   output/dataset/ur_locate_push2end_data/test/test_img_20251118/*_pose.json")


if __name__ == "__main__":
    main()
