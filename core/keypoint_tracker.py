"""
Keypoint Tracker Base Class Module
==================================

Base class for keypoint tracking functionality providing a common interface.
This base class defines the standard interface that all keypoint tracker implementations should follow.
"""

import os
import json
import time
import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from core.utils import get_project_paths


class KeypointTracker(ABC):
    """Base class for keypoint tracking implementations.
    
    This abstract base class defines the standard interface that all keypoint tracker 
    implementations should follow. It provides the same public interface as FFPPKeypointTracker
    to ensure consistency across different tracking methods.
    
    All keypoint tracker implementations should inherit from this class and implement
    the three core abstract methods:
    - set_reference_image(): Store reference image with keypoints
    - track_keypoints(): Track keypoints from reference to target image
    - remove_reference_image(): Remove stored reference images
    
    This design allows for different tracking backends (FlowFormer++, other optical flow
    methods, feature-based tracking, etc.) while maintaining a consistent interface.
    """
    
    def __init__(self, **kwargs):
        """Initialize the keypoint tracker base class.
        
        Args:
            **kwargs: Implementation-specific configuration parameters
        """
        # Common state that all implementations might need
        self.reference_data = {}
        self.default_reference_key = None
        self.processing_stats = {}
        
        # Get project paths for potential use by subclasses
        try:
            self.paths = get_project_paths()
        except Exception:
            # Fallback for when running standalone
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            self.paths = {
                'project_root': project_root,
                'output': os.path.join(project_root, 'output')
            }
    
    # ============================================================================
    # ABSTRACT METHODS - Must be implemented by all subclasses
    # ============================================================================
    
    @abstractmethod
    def set_reference_image(self, 
                          image: np.ndarray, 
                          keypoints: Optional[List[Dict]] = None,
                          image_name: Optional[str] = None) -> Dict:
        """Set reference image for keypoint tracking with optional image key.
        
        This is the first of three main public methods. Use this to store reference 
        images with their associated keypoints for later tracking operations.
        
        Args:
            image: Reference image as numpy array (H, W, 3) in RGB format.
            keypoints: Optional list of keypoint dictionaries with 'x', 'y' keys.
            image_name: Optional string name to identify this reference image. 
                       If None, uses 'default' and sets as default reference.
            
        Returns:
            Dict with success status and information about the set reference image.
            Should include at least:
            {
                'success': bool,
                'key': str,  # The key used to store this reference
                'keypoints_count': int,
                'is_default': bool,
                'error': str  # Only if success=False
            }
        """
        pass
    
    @abstractmethod
    def track_keypoints(self, 
                       target_image: np.ndarray,
                       reference_name: Optional[str] = None,
                       **kwargs) -> Dict:
        """Track keypoints from stored reference image to target image.
        
        This is the second of three main public methods. Use this to track keypoints
        from a stored reference image to a target image.
        
        Args:
            target_image: Target image as numpy array (H, W, 3) in RGB format.
            reference_name: Name of stored reference image to use. If None, uses default reference.
            **kwargs: Implementation-specific parameters (e.g., bidirectional=True, return_flow=True for FFPPKeypointTracker)
            
        Returns:
            Dict with tracking results. Should include at least:
            {
                'success': bool,
                'tracked_keypoints': List[Dict],  # List of tracked keypoint dicts with 'x', 'y'
                'keypoints_count': int,
                'processing_time': float,  # Total processing time in seconds
                'reference_name': str,
                'error': str  # Only if success=False
            }
        """
        pass
    
    @abstractmethod
    def remove_reference_image(self, image_name: Optional[str] = None) -> Dict:
        """Remove a stored reference image by name.
        
        This is the third of three main public methods. Use this to clean up
        stored reference images when they are no longer needed.
        
        Args:
            image_name: Name of the reference image to remove. If None, removes the default reference image.
            
        Returns:
            Dict with removal status. Should include at least:
            {
                'success': bool,
                'removed_key': str,  # The key that was actually removed
                'remaining_count': int,
                'error': str  # Only if success=False
            }
        """
        pass
    
    # ============================================================================
    # COMMON UTILITY METHODS - Available to all implementations
    # ============================================================================
    
    def get_reference_info(self) -> Dict:
        """Get information about currently stored reference images.
        
        Returns:
            Dict with reference image information:
            {
                'reference_count': int,
                'reference_keys': List[str],
                'default_key': Optional[str],
                'total_keypoints': Dict[str, int]  # key -> keypoint count mapping
            }
        """
        return {
            'reference_count': len(self.reference_data),
            'reference_keys': list(self.reference_data.keys()),
            'default_key': self.default_reference_key,
            'total_keypoints': {key: len(data.get('keypoints', [])) 
                               for key, data in self.reference_data.items()}
        }
    
    def clear_all_references(self) -> Dict:
        """Clear all stored reference images.
        
        Returns:
            Dict with clearing status:
            {
                'success': bool,
                'cleared_count': int,
                'cleared_keys': List[str]
            }
        """
        cleared_keys = list(self.reference_data.keys())
        cleared_count = len(cleared_keys)
        
        self.reference_data.clear()
        self.default_reference_key = None
        
        return {
            'success': True,
            'cleared_count': cleared_count,
            'cleared_keys': cleared_keys
        }
    
    def get_processing_stats(self) -> Dict:
        """Get the latest processing statistics.
        
        Returns:
            Dict with processing statistics from the last operation
        """
        return self.processing_stats.copy()
    
    def _validate_image(self, image: np.ndarray, name: str = "image") -> Optional[str]:
        """Validate image format and return error message if invalid.
        
        Args:
            image: Image array to validate
            name: Name of the image for error messages
            
        Returns:
            None if valid, error message string if invalid
        """
        if not isinstance(image, np.ndarray):
            return f'{name} must be numpy array, got {type(image)}'
        
        if image.ndim != 3 or image.shape[2] != 3:
            return f'{name} must be RGB format (H, W, 3), got {image.shape}'
        
        if image.size == 0:
            return f'{name} cannot be empty'
        
        return None
    
    def _validate_keypoints(self, keypoints: List[Dict]) -> Optional[str]:
        """Validate keypoints format and return error message if invalid.
        
        Args:
            keypoints: List of keypoint dictionaries to validate
            
        Returns:
            None if valid, error message string if invalid
        """
        if not isinstance(keypoints, list):
            return f'Keypoints must be list, got {type(keypoints)}'
        
        for i, kp in enumerate(keypoints):
            if not isinstance(kp, dict):
                return f'Keypoint {i} must be dict, got {type(kp)}'
            
            if 'x' not in kp or 'y' not in kp:
                return f'Keypoint {i} must have "x" and "y" keys, got {list(kp.keys())}'
            
            try:
                float(kp['x'])
                float(kp['y'])
            except (ValueError, TypeError):
                return f'Keypoint {i} coordinates must be numeric, got x={kp["x"]}, y={kp["y"]}'
        
        return None
