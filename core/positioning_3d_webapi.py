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
