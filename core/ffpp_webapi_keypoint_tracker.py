"""
FlowFormer++ Web API Keypoint Tracker Module
===========================================

Web API client for FlowFormer++ keypoint tracking service.

This module provides the FFPPWebAPIKeypointTracker class that interfaces with
the FlowFormer++ Flask web service for keypoint tracking. It maintains the same
interface as the direct FFPPKeypointTracker but uses HTTP API calls instead of
direct model inference.

Features:
- Implements the standard KeypointTracker interface for consistency
- HTTP client for FlowFormer++ Flask service
- File upload handling for images
- JSON-based communication with service
- Same three-method interface (set_reference_image + track_keypoints + remove_reference_image)
- Error handling and service availability checking
- Compatible with existing code that uses FFPPKeypointTracker

Usage:
    from core.ffpp_webapi_keypoint_tracker import FFPPWebAPIKeypointTracker
    
    tracker = FFPPWebAPIKeypointTracker(service_url="http://localhost:8001")
    tracker.set_reference_image(ref_image, keypoints)
    result = tracker.track_keypoints(target_image)
    tracker.remove_reference_image()  # Clean up when done

The web API approach is useful for:
- Distributed architectures where tracking runs on a separate server
- Avoiding model loading overhead for occasional tracking tasks
- Microservices architectures
- Remote tracking capabilities
"""

import os
import sys
import time
import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
import io

# Import the base class
from core.keypoint_tracker import KeypointTracker

# Conditional imports to avoid issues when running as main
try:
    from core.utils import get_project_paths
except ImportError:
    # When running as main, imports will be handled in main function
    def get_project_paths():
        """Fallback function for get_project_paths when utils not available."""
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        return {
            'project_root': project_root,
            'output': project_root / 'output',
            'sample_data': project_root / 'sample_data'
        }


class FFPPWebAPIKeypointTracker(KeypointTracker):
    """FlowFormer++ Web API Keypoint Tracker.
    
    This class provides keypoint tracking functionality by communicating with
    a FlowFormer++ Flask web service via HTTP API calls. It maintains the same
    interface as FFPPKeypointTracker but uses web API instead of direct model inference.
    
    The tracker communicates with a Flask service that should provide these endpoints:
    - GET /health - Service health check
    - GET /references - List stored reference images
    - POST /set_reference - Set reference image with keypoints
    - POST /track_keypoints - Track keypoints in target image
    
    Attributes:
        service_url (str): Base URL of the FlowFormer++ web service
        timeout (int): Request timeout in seconds
        session (requests.Session): HTTP session for connection pooling
    """
    
    def __init__(self, 
                 service_url: str = "http://localhost:8001",
                 timeout: int = 30,
                 **kwargs):
        """Initialize the FlowFormer++ Web API keypoint tracker.
        
        Args:
            service_url: Base URL of the FlowFormer++ web service (default: http://localhost:8001)
            timeout: Request timeout in seconds (default: 30)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Test service availability
        self._check_service_availability()
    
    # Public Interface Methods (required by KeypointTracker base class)
    
    def set_reference_image(self, 
                           image: np.ndarray,
                           keypoints: Optional[List[Dict]] = None, 
                           image_name: Optional[str] = None) -> Dict:
        """Set reference image with keypoints using the web API.
        
        Args:
            image: Reference image as numpy array (H, W, 3) in RGB format
            keypoints: List of keypoint dictionaries with 'x', 'y' keys
            image_name: Optional name for this reference image
            
        Returns:
            Dict with success status and reference image information
        """
        try:
            # Validate inputs
            if keypoints is None:
                keypoints = []
            
            if not isinstance(keypoints, list):
                return {
                    'success': False,
                    'error': 'Keypoints must be a list of dictionaries'
                }
            
            # Prepare image data
            image_bytes = self._numpy_to_pil_bytes(image)
            
            # Prepare form data
            files = {
                'image': ('reference.png', image_bytes, 'image/png')
            }
            
            data = {
                'keypoints': json.dumps(keypoints)
            }
            
            if image_name:
                data['image_name'] = image_name
            
            # Make API call
            response = self.session.post(
                f"{self.service_url}/set_reference",
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    # Update local reference tracking
                    ref_key = result.get('data', {}).get('image_name', 'default')
                    self.reference_data[ref_key] = {
                        'keypoints_count': len(keypoints),
                        'set_time': time.time()
                    }
                    
                    if image_name is None or ref_key == 'default':
                        self.default_reference_key = ref_key
                    
                    return {
                        'success': True,
                        'key': ref_key,
                        'keypoints_count': len(keypoints),
                        'is_default': ref_key == self.default_reference_key,
                        'service_response': result.get('data', {})
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Service error: {result.get('message', 'Unknown error')}"
                    }
            else:
                return {
                    'success': False,
                    'error': f"HTTP error {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            }
    
    def track_keypoints(self, 
                       target_image: np.ndarray,
                       reference_name: Optional[str] = None,
                       bidirectional: bool = False,
                       **kwargs) -> Dict:
        """Track keypoints from reference to target image using the web API.
        
        Args:
            target_image: Target image as numpy array (H, W, 3) in RGB format
            reference_name: Name of reference image to use (None for default)
            bidirectional: Enable bidirectional validation (default: False)
            **kwargs: Additional arguments (ignored for web API)
            
        Returns:
            Dict with tracking results including tracked keypoints and statistics
        """
        try:
            # Check if we have reference images
            if not self.reference_data:
                return {
                    'success': False,
                    'error': 'No reference images set. Call set_reference_image() first.'
                }
            
            # Prepare image data
            image_bytes = self._numpy_to_pil_bytes(target_image)
            
            # Prepare form data
            files = {
                'image': ('target.png', image_bytes, 'image/png')
            }
            
            data = {
                'bidirectional': 'true' if bidirectional else 'false'
            }
            
            if reference_name:
                data['reference_name'] = reference_name
            
            # Make API call
            response = self.session.post(
                f"{self.service_url}/track_keypoints",
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    service_data = result.get('data', {})
                    
                    # Extract tracking results
                    tracked_keypoints = service_data.get('tracked_keypoints', [])
                    
                    return {
                        'success': True,
                        'tracked_keypoints': tracked_keypoints,
                        'keypoints_count': len(tracked_keypoints),
                        'reference_name': service_data.get('reference_used'),
                        'total_processing_time': service_data.get('processing_time', 0),
                        'bidirectional_enabled': bidirectional,
                        'bidirectional_stats': service_data.get('bidirectional_stats'),
                        'device_used': service_data.get('device_used', 'unknown'),
                        'service_response': service_data
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Service error: {result.get('message', 'Unknown error')}"
                    }
            else:
                return {
                    'success': False,
                    'error': f"HTTP error {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            }
    
    def remove_reference_image(self, image_name: Optional[str] = None) -> Dict:
        """Remove a stored reference image by name.
        
        Note: This web API client doesn't directly control server-side reference storage.
        It only manages local reference tracking. The actual removal would need to be
        implemented on the server side.
        
        Args:
            image_name: Name of reference image to remove (None for default)
            
        Returns:
            Dict with removal status
        """
        try:
            # Determine which reference to remove
            if image_name is None:
                if self.default_reference_key:
                    key_to_remove = self.default_reference_key
                else:
                    return {
                        'success': False,
                        'error': 'No default reference image to remove'
                    }
            else:
                key_to_remove = image_name
            
            # Check if reference exists locally
            if key_to_remove not in self.reference_data:
                return {
                    'success': False,
                    'error': f'Reference image "{key_to_remove}" not found'
                }
            
            # Remove from local tracking
            del self.reference_data[key_to_remove]
            
            # Update default reference if necessary
            if key_to_remove == self.default_reference_key:
                if self.reference_data:
                    self.default_reference_key = next(iter(self.reference_data.keys()))
                else:
                    self.default_reference_key = None
            
            return {
                'success': True,
                'removed_key': key_to_remove,
                'remaining_count': len(self.reference_data),
                'note': 'Local reference tracking updated. Server-side storage may still contain the reference.'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            }
    
    # Additional Public Methods (service-specific utilities)
    
    def get_service_references(self) -> Dict:
        """Get reference images stored on the server.
        
        Returns:
            Dict with server-side reference information
        """
        try:
            response = self.session.get(
                f"{self.service_url}/references",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    data = result.get('data', {})
                    return {
                        'success': True,
                        'server_references': data.get('references', {}),
                        'default_reference': data.get('default_reference'),
                        'total_count': data.get('total_count', 0)
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Service error: {result.get('message', 'Unknown error')}"
                    }
            else:
                return {
                    'success': False,
                    'error': f"HTTP error {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            }
    
    def get_service_health(self) -> Dict:
        """Get health status of the FlowFormer++ web service.
        
        Returns:
            Dict with service health information
        """
        try:
            response = self.session.get(
                f"{self.service_url}/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'success': False,
                    'error': f"HTTP error {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}"
            }
    
    def __del__(self):
        """Clean up the HTTP session."""
        if hasattr(self, 'session'):
            self.session.close()
    
    # Private Methods (internal implementation details)
    
    def _check_service_availability(self):
        """Check if the FlowFormer++ web service is available."""
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('success', False):
                    print(f"✅ FlowFormer++ Web API service available at {self.service_url}")
                    return True
                else:
                    print(f"⚠️ FlowFormer++ Web API service degraded: {health_data.get('message', 'Unknown status')}")
                    return False
            else:
                print(f"❌ FlowFormer++ Web API service returned HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to FlowFormer++ Web API service at {self.service_url}: {e}")
            return False
    
    def _numpy_to_pil_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy array to PIL Image bytes for upload.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Image data as bytes in PNG format
        """
        # Ensure image is in the correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image, 'RGB')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()


def main():
    """Example usage of the FFPPWebAPIKeypointTracker."""
    import numpy as np
    from pathlib import Path
    
    print("🔗 FlowFormer++ Web API Keypoint Tracker Test")
    print("=" * 50)
    
    # Initialize tracker
    tracker = FFPPWebAPIKeypointTracker()
    
    # Check service health
    health = tracker.get_service_health()
    print(f"Service Health: {health}")
    
    # Create test data
    ref_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    target_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    keypoints = [
        {'x': 100, 'y': 150},
        {'x': 200, 'y': 200},
        {'x': 300, 'y': 100}
    ]
    
    print("\n🎯 Testing API tracking workflow...")
    
    # Set reference image
    print("1. Setting reference image...")
    result = tracker.set_reference_image(ref_image, keypoints, "test_ref")
    print(f"   Result: {result.get('success')} - {result.get('keypoints_count', 0)} keypoints")
    
    if result.get('success'):
        # Track keypoints
        print("2. Tracking keypoints...")
        result = tracker.track_keypoints(target_image, bidirectional=True)
        print(f"   Result: {result.get('success')} - {result.get('keypoints_count', 0)} tracked")
        print(f"   Processing time: {result.get('total_processing_time', 0):.2f}s")
        
        # Check server references
        print("3. Checking server references...")
        refs = tracker.get_service_references()
        print(f"   Server references: {refs}")
        
        # Clean up
        print("4. Removing reference...")
        result = tracker.remove_reference_image("test_ref")
        print(f"   Result: {result}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()