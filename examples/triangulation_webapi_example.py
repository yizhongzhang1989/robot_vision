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
    
    def init_session(self, reference_name: str) -> Dict:
        """
        Initialize a new positioning session.
        
        Args:
            reference_name: Name of the reference image to use
            
        Returns:
            Response with session_id if successful
        """
        try:
            payload = {
                'reference_name': reference_name
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
    
    def upload_view(self, session_id: str, image: np.ndarray,
                   intrinsic: np.ndarray, distortion: Optional[np.ndarray],
                   extrinsic: np.ndarray) -> Dict:
        """
        Upload a camera view for triangulation.
        
        Args:
            session_id: Session identifier from init_session
            image: Camera image as numpy array (BGR or RGB)
            intrinsic: 3x3 camera intrinsic matrix
            distortion: Distortion coefficients (optional, can be None)
            extrinsic: 4x4 camera extrinsic matrix (world to camera)
            
        Returns:
            Response with queue position if successful
        """
        try:
            # Convert image to base64
            image_base64 = image_to_base64(image)
            
            # Get image size from array
            h, w = image.shape[:2]
            image_size = (w, h)
            
            # Generate unique view_id
            view_id = f"view_{int(time.time() * 1000000)}"
            
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
    
    def get_result(self, session_id: str, timeout: int = 0) -> Dict:
        """
        Get triangulation result for a session.
        
        Args:
            session_id: Session identifier
            timeout: Maximum wait time in milliseconds. If 0, return immediately.
                    If > 0, wait up to timeout ms for completion.
            
        Returns:
            Triangulation result with 3D points and per-view keypoints.
            If not completed and timeout=0, returns session status instead.
        """
        try:
            if timeout == 0:
                # Check if completed first
                status_response = self.session.get(f"{self.service_url}/session_status/{session_id}")
                status_response.raise_for_status()
                status_data = status_response.json()
                
                if not status_data.get('success'):
                    return status_data
                
                session_info = status_data.get('session', {})
                session_status = session_info.get('status')
                
                if session_status != 'completed':
                    # Return status if not completed
                    return status_data
                
                # Get result if completed
                response = self.session.get(f"{self.service_url}/result/{session_id}")
                response.raise_for_status()
                return response.json()
            else:
                # Wait for completion with timeout
                start_time = time.time() * 1000  # Convert to ms
                check_interval = 100  # ms
                
                while True:
                    elapsed = (time.time() * 1000) - start_time
                    
                    if elapsed >= timeout:
                        # Timeout - return current status
                        status_response = self.session.get(f"{self.service_url}/session_status/{session_id}")
                        status_response.raise_for_status()
                        status_data = status_response.json()
                        if status_data.get('success'):
                            status_data['timeout'] = True
                        return status_data
                    
                    # Check status
                    status_response = self.session.get(f"{self.service_url}/session_status/{session_id}")
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    
                    if not status_data.get('success'):
                        return status_data
                    
                    session_info = status_data.get('session', {})
                    session_status = session_info.get('status')
                    
                    if session_status == 'completed':
                        # Get result
                        response = self.session.get(f"{self.service_url}/result/{session_id}")
                        response.raise_for_status()
                        return response.json()
                    elif session_status == 'failed':
                        # Return failure status
                        return status_data
                    
                    # Wait before next check
                    time.sleep(check_interval / 1000.0)
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def terminate_session(self, session_id: str) -> Dict:
        """
        Terminate a session and remove its data from the server.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Response with success status
        """
        try:
            response = self.session.post(f"{self.service_url}/terminate_session/{session_id}")
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
        image: Image as numpy array (BGR format from OpenCV)
        
    Returns:
        Base64 encoded image string with data URI prefix
        
    Note:
        The image is encoded directly as PNG without color space conversion.
        The FFPP server will handle any necessary conversions internally.
    """
    # Encode as PNG directly (no color conversion needed)
    success, buffer = cv2.imencode('.png', image)
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
        where extrinsic is world2cam (base2cam) for triangulation
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
    
    # Calculate cam2base (camera to world/base)
    cam2base = end2base @ cam2end
    
    # Triangulation expects world2cam (world to camera), so invert
    # base2cam = inv(cam2base)
    extrinsic = np.linalg.inv(cam2base)
    
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
    test_data_dir = Path("dataset/ur_locate_push2end_data/test/test_img_20251118")
    reference_name = "ur_locate_push2end_data"
    
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
    print(f"\n4. Initializing session...")
    session_result = client.init_session(
        reference_name=reference_name
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
        print(f"   Uploading {img_file.name}...", end=" ")
        
        result = client.upload_view(
            session_id=session_id,
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
    
    # Clean up: terminate session
    print("\n8. Terminating session...")
    term_result = client.terminate_session(session_id)
    if term_result.get('success'):
        print("✅ Session terminated and cleaned up")
    else:
        print(f"⚠️  Failed to terminate session: {term_result.get('error')}")
    
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
        print("   dataset/ref_name/ref_img_1.jpg")
        print("   dataset/ref_name/ref_img_1.json")
        print("\n4. Verify test images exist:")
        print("   dataset/ur_locate_push2end_data/test/test_img_20251118/*.jpg")
        print("   dataset/ur_locate_push2end_data/test/test_img_20251118/*_pose.json")


if __name__ == "__main__":
    main()
