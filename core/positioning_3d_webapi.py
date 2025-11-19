#!/usr/bin/env python3
"""
3D Positioning Web API Client
==============================

Client library for interacting with the 3D Positioning Service Web API.
Provides convenient methods for multi-view triangulation with FlowFormer++
keypoint tracking.

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


class Positioning3DWebAPIClient:
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
    
    def get_result(self, session_id: str, timeout: int = 30000) -> Dict:
        """
        Get triangulation result for a session.
        
        Triggers on-demand triangulation if not already completed.
        Server will wait for pending views to be tracked before triangulation.
        
        Args:
            session_id: Session identifier
            timeout: Maximum wait time in milliseconds for view tracking and triangulation.
                    Default: 30000ms (30 seconds)
            
        Returns:
            Triangulation result with 3D points and per-view keypoints.
        """
        try:
            # Call server endpoint with timeout parameter
            # Server will trigger triangulation and wait for tracking
            response = self.session.get(
                f"{self.service_url}/result/{session_id}",
                params={'timeout': timeout}
            )
            response.raise_for_status()
            return response.json()
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_result_view_plane(self, session_id: str, plane_point: np.ndarray,
                             plane_normal: np.ndarray, timeout: int = 0) -> Dict:
        """
        Get view-plane triangulation result for a single-view session.
        
        Projects 2D points from a single camera view onto a known 3D plane
        by intersecting camera rays with the plane.
        
        Args:
            session_id: Session identifier
            plane_point: 3D point on the plane in world coordinates (shape: (3,))
            plane_normal: Normal vector of the plane (shape: (3,))
            timeout: Maximum wait time in milliseconds. If 0, return immediately.
                    If > 0, wait up to timeout ms for view to be tracked.
            
        Returns:
            View-plane triangulation result with 3D points on the plane.
            If not completed and timeout=0, returns session status instead.
            If more than one view uploaded, returns error.
        """
        try:
            # First check session status
            status_response = self.session.get(f"{self.service_url}/session_status/{session_id}")
            status_response.raise_for_status()
            status_data = status_response.json()
            
            if not status_data.get('success'):
                return status_data
            
            session_info = status_data.get('session', {})
            session_status = session_info.get('status')
            progress = session_info.get('progress', {})
            views_received = progress.get('views_received', 0)
            
            # Check that exactly 1 view is uploaded
            if views_received == 0:
                return {
                    'success': False,
                    'error': 'No views uploaded for view-plane triangulation',
                    'session': session_info
                }
            elif views_received > 1:
                return {
                    'success': False,
                    'error': f'View-plane triangulation requires exactly 1 view, but {views_received} views were uploaded',
                    'session': session_info
                }
            
            # Wait for tracking to complete if timeout specified
            if timeout > 0:
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
                    
                    # Check if tracking is complete
                    status_response = self.session.get(f"{self.service_url}/session_status/{session_id}")
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    
                    if not status_data.get('success'):
                        return status_data
                    
                    session_info = status_data.get('session', {})
                    progress = session_info.get('progress', {})
                    views_tracked = progress.get('views_tracked', 0)
                    
                    if views_tracked >= 1:
                        # Tracking complete, proceed to triangulation
                        break
                    
                    # Wait before next check
                    time.sleep(check_interval / 1000.0)
            
            # Call view-plane triangulation endpoint
            payload = {
                'session_id': session_id,
                'plane_point': plane_point.tolist(),
                'plane_normal': plane_normal.tolist()
            }
            
            response = self.session.post(
                f"{self.service_url}/result_view_plane",
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
        
    Raises:
        ValueError: If JSON is malformed or missing required fields
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e.msg} at line {e.lineno}")
    except Exception as e:
        raise ValueError(f"Failed to read {json_path}: {str(e)}")
    
    # Validate required fields
    required_fields = ['camera_matrix', 'distortion_coefficients', 'end2base', 'cam2end_matrix']
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields in {json_path}: {', '.join(missing_fields)}")
    
    try:
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
        
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse camera parameters from {json_path}: {str(e)}")
