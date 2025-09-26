"""
FlThis module provides the FFPPWebAPIKeypointTracker class that interfaces with
the FlowFormer++ Flask web service for keypoint tracking. It maintains the same
interface as the direct FFPPKeypointTracker but uses HTTP API calls instead of
direct model inference.

Features:
- Implements the standard KeypointTracker interface for consistency
- HTTP client for FlowFormer++ Flask service
- Compressed JPG with base64 encoding for efficient image transmission
- Pure JSON-based communication with service (no multipart forms)
- Same three-method interface (set_reference_image + track_keypoints + remove_reference_image)
- Error handling and service availability checking
- Compatible with existing code that uses FFPPKeypointTrackerb API Keypoint Tracker Module
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
import base64
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
    a FlowFormer++ Flask web service via HTTP API calls. It maintains the exact same
    interface as FFPPKeypointTracker but uses remote web API instead of direct model inference.
    
    The web API approach enables:
    - Distributed architectures where tracking runs on a separate server
    - Avoiding model loading overhead for occasional tracking tasks  
    - Microservices architectures with centralized GPU resources
    - Remote tracking capabilities across network boundaries
    - Easier deployment without local model dependencies
    
    Key Interface Methods:
        set_reference_image(): Upload reference image and keypoints to web service
        track_keypoints():     Track keypoints via API call with HTTP image upload
        remove_reference_image(): Remove reference from local tracking
    
    Web Service Specific Methods:
        get_service_health():     Check web service status and availability
        get_service_references(): List all references stored on server
    
    Required Web Service Endpoints:
        GET    /health                - Service health check and status
        GET    /references            - List stored reference images with metadata
        POST   /set_reference_image   - Set reference image with keypoints (JSON with base64 image)
        POST   /track_keypoints       - Track keypoints in target image (JSON with base64 image)
        POST   /remove_reference_image - Remove reference image by name (JSON with image_name)
    
    Attributes:
        service_url (str): Base URL of the FlowFormer++ web service (default: http://localhost:8001)
        timeout (int): Request timeout in seconds for HTTP operations (default: 30s)
        session (requests.Session): HTTP session with connection pooling for efficiency
        reference_data (dict): Local tracking of references for interface consistency
        default_reference_key (str): Local default reference key for interface consistency
        
    Example:
        tracker = FFPPWebAPIKeypointTracker(service_url="http://gpu-server:8001")
        tracker.set_reference_image(ref_image, keypoints, "scene1")
        result = tracker.track_keypoints(target_image, bidirectional=True)
        
    Note:
        Requires FlowFormer++ web service running and accessible at service_url.
        Network errors, service errors, and timeouts are handled gracefully.
        Local reference tracking maintains consistency with base KeypointTracker interface.
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
    
    # ============================================================================
    # PUBLIC INTERFACE METHODS (required by KeypointTracker base class)
    # ============================================================================
    
    def set_reference_image(self, 
                           image: np.ndarray,
                           keypoints: Optional[List[Dict]] = None, 
                           image_name: Optional[str] = None) -> Dict:
        """Set reference image for keypoint tracking via web API with optional image key.
        
        This is the first of two main public methods. Use this to store reference 
        images with their associated keypoints on the web service for later tracking operations.
        The image and keypoints are uploaded to the FlowFormer++ web service via HTTP JSON API.
        
        Args:
            image: Reference image as numpy array (H, W, 3) in RGB format.
                  Image will be compressed to JPG and base64 encoded for JSON transmission.
            keypoints: Optional list of keypoint dictionaries with 'x', 'y' keys.
                      If None, defaults to empty list. Each keypoint should be a dict
                      like {'x': float, 'y': float} in image pixel coordinates.
            image_name: Optional string name to identify this reference image on server.
                       If None, server will assign 'default' and set as default reference.
            
        Returns:
            Dict with success status and information about the set reference image.
            On success, includes 'key', 'keypoints_count', 'is_default', and 'service_response'.
            On failure, includes 'error' with detailed error message.
            
        Note:
            The image is automatically converted to compressed JPG and base64 encoded for JSON.
            This provides efficient network transfer with good image quality (85% JPG quality).
            Network errors, service errors, and validation errors are handled gracefully.
            Local reference tracking is updated to maintain consistency with base class interface.
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
            
            # Prepare JSON data with base64 encoded image
            image_base64 = self._numpy_to_base64_jpg(image)
            
            data = {
                'image_base64': image_base64,
                'keypoints': keypoints
            }
            
            if image_name:
                data['image_name'] = image_name
            
            # Make API call with JSON using tracker-compatible endpoint name
            response = self.session.post(
                f"{self.service_url}/set_reference_image",
                json=data,
                headers={'Content-Type': 'application/json'},
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
        """Track keypoints from stored reference image to target image via web API.
        
        This is the second of two main public methods. Use this to track keypoints
        from a stored reference image to a target image using the FlowFormer++ web service.
        The target image is uploaded via HTTP API and processed by the remote service.
        
        Args:
            target_image: Target image as numpy array (H, W, 3) in RGB format.
                         Image will be compressed to JPG and base64 encoded for JSON transmission.
            reference_name: Name of stored reference image to use on server. If None, uses default reference.
                           Must match a reference name previously set via set_reference_image().
            bidirectional: If True, request bidirectional flow validation from service. Tracks keypoints
                          forward (ref→target) then backward (target→ref) and measures consistency.
                          Default is False for faster processing.
            **kwargs: Additional arguments (ignored for web API compatibility with direct tracker).
            
        Returns:
            Dict: Tracking results with success status, tracked keypoints, and statistics.
                 On success, includes 'tracked_keypoints', 'keypoints_count', 'total_processing_time',
                 'device_used', and 'service_response'. If bidirectional=True, includes 
                 'bidirectional_stats' with accuracy metrics for each keypoint.
                 On failure, includes 'error' with detailed error message.
        
        Note:
            You must call set_reference_image() first to store a reference image with keypoints
            on the web service before using this function. The target image is automatically
            compressed to JPG and base64 encoded for efficient JSON transmission. Network errors 
            and service errors are handled gracefully with detailed error reporting.
            
            Bidirectional mode: Service computes ref→target flow, then target→ref flow to validate
            tracking accuracy. Returns consistency distance for each keypoint via 'bidirectional_stats'.
        """
        try:
            # Check if we have reference images
            if not self.reference_data:
                return {
                    'success': False,
                    'error': 'No reference images set. Call set_reference_image() first.'
                }
            
            # Prepare JSON data with base64 encoded image
            image_base64 = self._numpy_to_base64_jpg(target_image)
            
            data = {
                'image_base64': image_base64,
                'bidirectional': bidirectional
            }
            
            if reference_name:
                data['reference_name'] = reference_name
            
            # Make API call with JSON using tracker-compatible endpoint name
            response = self.session.post(
                f"{self.service_url}/track_keypoints",
                json=data,
                headers={'Content-Type': 'application/json'},
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
        """Remove a stored reference image by name via web API call.
        
        This method removes references from both local tracking and server-side storage
        by calling the web service's remove_reference_image endpoint.
        
        Args:
            image_name: Name of the reference image to remove. If None, removes the default reference image.
                       Must match a reference name previously set via set_reference_image().
            
        Returns:
            Dict with success status and information about the removal operation.
            On success, includes 'removed_key', 'remaining_count', and server response.
            On failure, includes 'error' with detailed error message.
            
        Note:
            This method now calls the web service to remove references from server-side storage
            and updates local tracking accordingly. Both client and server state are synchronized.
            Network errors and service errors are handled gracefully with detailed error reporting.
        """
        try:
            # Determine which reference to remove locally first
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
                    'error': f'Reference image "{key_to_remove}" not found in local tracking'
                }
            
            # Call web service to remove from server
            data = {'image_name': key_to_remove}
            
            response = self.session.post(
                f"{self.service_url}/remove_reference_image",
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    # Remove from local tracking on successful server removal
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
                        'server_response': result.get('data', {}),
                        'note': 'Reference removed from both local tracking and server storage.'
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Server error: {result.get('message', 'Unknown error')}"
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
    
    # ============================================================================
    # ADDITIONAL PUBLIC METHODS (web service specific utilities)
    # ============================================================================
    
    def get_service_references(self) -> Dict:
        """Get reference images stored on the web service server.
        
        This web API specific method queries the FlowFormer++ service to retrieve
        information about all reference images currently stored on the server.
        Useful for debugging, monitoring, and understanding server state.
        
        Returns:
            Dict with server-side reference information and metadata.
            On success, includes 'server_references' (dict of reference info),
            'default_reference' (server's default reference name), and 'total_count'.
            On failure, includes 'error' with detailed error message.
            
        Note:
            Server references may differ from local reference tracking due to:
            - References set by other clients
            - Server persistence across client sessions  
            - Network issues during reference operations
            This method provides ground truth of server state.
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
        """Get health status and diagnostics of the FlowFormer++ web service.
        
        This web API specific method performs a health check on the remote service,
        providing real-time status information for monitoring and troubleshooting.
        Useful for validating service availability before tracking operations.
        
        Returns:
            Dict with comprehensive service health information.
            On success, returns the full service health response including 'status',
            'message', 'success', and any additional service-specific metadata.
            On failure, includes 'error' with detailed network or service error information.
            
        Note:
            This is a lightweight operation that tests basic service connectivity
            and provides service status. Use this method to verify service availability
            before attempting tracking operations, especially in distributed environments.
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
    
    # ============================================================================
    # PRIVATE METHODS (internal implementation details)
    # ============================================================================
    
    def _check_service_availability(self):
        """Check if the FlowFormer++ web service is available and responsive.
        
        Internal method called during initialization to verify service connectivity.
        Performs a quick health check with short timeout to validate the service URL.
        Provides user feedback about service status during tracker initialization.
        
        Returns:
            bool: True if service is available and healthy, False otherwise.
            
        Note:
            This method prints status messages directly to stdout for immediate user feedback.
            Uses a short 5-second timeout for quick connectivity testing during init.
            Network errors and service errors are caught and reported gracefully.
        """
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
    
    def _numpy_to_base64_jpg(self, image: np.ndarray, quality: int = 85) -> str:
        """Convert numpy array to base64 encoded JPG string for JSON transmission.
        
        Internal method for preparing compressed image data for web API JSON transmission.
        Handles data type conversion and JPG compression for efficient network transfer.
        
        Args:
            image: RGB image as numpy array (H, W, 3) in any numeric dtype.
                  Values are automatically normalized to uint8 range if needed.
            quality: JPG compression quality (1-100, default 85). Higher = better quality, larger size.
            
        Returns:
            str: Image data encoded as base64 JPG string suitable for JSON transmission.
                 Uses JPG compression for efficient network transfer with good quality.
                   
        Note:
            Automatically converts float images to uint8 by scaling [0,1] → [0,255].
            Uses PIL for reliable JPG encoding with customizable compression quality.
            JPG format chosen for compact size with acceptable quality loss for API transmission.
            Base64 encoding allows binary data to be included in JSON payloads.
        """
        # Ensure image is in the correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image, 'RGB')
        
        # Convert to JPG bytes with compression
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG', quality=quality)
        img_bytes.seek(0)
        
        # Encode to base64 string
        image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        return image_base64

