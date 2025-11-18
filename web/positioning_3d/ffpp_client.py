"""
FFPP Client for 3D Positioning Service
======================================

Handles communication with FlowFormer++ Keypoint Tracking Service.
"""

import requests
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)


class FFPPClient:
    """Client for FlowFormer++ Keypoint Tracking Service."""
    
    def __init__(self, host: str = "localhost", port: int = 8001, timeout: int = 30):
        """
        Initialize FFPP client.
        
        Args:
            host: FFPP server hostname or IP
            port: FFPP server port
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        
        logger.info(f"FFPP Client initialized: {self.base_url}")
    
    def health_check(self) -> bool:
        """
        Check if FFPP server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"FFPP health check: {data.get('message', 'OK')}")
                return data.get('success', False)
            else:
                logger.warning(f"FFPP health check failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"FFPP health check error: {e}")
            return False
    
    def set_reference_image(
        self,
        image: np.ndarray,
        keypoints: List[Dict[str, float]],
        image_name: str
    ) -> Dict[str, Any]:
        """
        Upload reference image with keypoints to FFPP server.
        
        Args:
            image: Image as numpy array (RGB)
            keypoints: List of keypoints [{'x': float, 'y': float}, ...]
            image_name: Reference image name
            
        Returns:
            Response dict with 'success' and optional 'error' fields
        """
        try:
            # Convert numpy array to base64
            image_base64 = self._numpy_to_base64(image)
            
            # Prepare request
            payload = {
                'image_base64': image_base64,
                'keypoints': keypoints,
                'image_name': image_name
            }
            
            # Send request
            response = requests.post(
                f"{self.base_url}/set_reference_image",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    logger.info(f"Reference image '{image_name}' uploaded successfully")
                else:
                    logger.error(f"Failed to upload reference '{image_name}': {data.get('message')}")
                return data
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Failed to upload reference '{image_name}': {error_msg}")
                return {
                    'success': False,
                    'error': error_msg
                }
                
        except Exception as e:
            logger.error(f"Error uploading reference '{image_name}': {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def track_keypoints(
        self,
        image_base64: str,
        reference_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track keypoints in target image using pre-set reference.
        
        Args:
            image_base64: Target image as base64 string
            reference_name: Name of reference image (optional, uses default if None)
            
        Returns:
            Response dict with 'success', 'result' (tracked_keypoints), or 'error'
        """
        try:
            # Prepare request
            payload = {
                'image_base64': image_base64,
                'bidirectional': True,  # Use bidirectional flow for better accuracy
                'return_flow': False
            }
            
            if reference_name:
                payload['reference_name'] = reference_name
            
            # Send request
            response = requests.post(
                f"{self.base_url}/track_keypoints",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    tracked_kps = data.get('result', {}).get('tracked_keypoints', [])
                    logger.debug(f"Tracked {len(tracked_kps)} keypoints")
                else:
                    logger.warning(f"Tracking failed: {data.get('message')}")
                return data
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Tracking request failed: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg
                }
                
        except Exception as e:
            logger.error(f"Error tracking keypoints: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_references(self) -> Dict[str, Any]:
        """
        List all reference images stored on FFPP server.
        
        Returns:
            Response dict with 'success' and 'result' containing references
        """
        try:
            response = requests.get(
                f"{self.base_url}/references",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error listing references: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _numpy_to_base64(self, image: np.ndarray) -> str:
        """
        Convert numpy array to base64 encoded PNG string.
        
        Args:
            image: Image as numpy array (RGB)
            
        Returns:
            Base64 encoded PNG string with data URL prefix
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Save to bytes buffer as PNG
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Add data URL prefix
        return f"data:image/png;base64,{image_base64}"
    
    def retry_request(
        self,
        request_func,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retry a request with exponential backoff.
        
        Args:
            request_func: Function to call (should return dict with 'success')
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff multiplier for retry delays
            **kwargs: Arguments to pass to request_func
            
        Returns:
            Response dict from request_func
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = request_func(**kwargs)
                
                if result.get('success'):
                    return result
                    
                last_error = result.get('error', 'Unknown error')
                
                if attempt < max_retries:
                    delay = backoff_factor ** attempt
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s...")
                    time.sleep(delay)
                    
            except Exception as e:
                last_error = str(e)
                
                if attempt < max_retries:
                    delay = backoff_factor ** attempt
                    logger.warning(f"Request exception (attempt {attempt + 1}/{max_retries + 1}): {e}, retrying in {delay}s...")
                    time.sleep(delay)
        
        return {
            'success': False,
            'error': f"Failed after {max_retries + 1} attempts: {last_error}"
        }
