"""
FlowFormer++ Keypoint Tracker Module
===================================

High-performance keypoint tracking module using direct FlowFormer++ model integration.

This module provides the FFPPKeypointTracker class for efficient keypoint tracking
with features including:
- 21x faster tracking compared to API-based solutions
- Simplified two-step interface (set_reference_image + track_keypoints)
- Bidirectional flow validation for accuracy assessment
- Multiple reference image management
- GPU acceleration with CUDA support

Usage:
    from core.ffpp_keypoint_tracker import FFPPKeypointTracker
    
    tracker = FFPPKeypointTracker()
    tracker.set_reference_image(ref_image, keypoints)
    result = tracker.track_keypoints(target_image)

For examples and test cases, see: examples/ffpp_keypoint_tracker_example.py
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image

# Conditional imports to avoid issues when running as main
try:
    from core.utils import (load_keypoints, resize_keypoints, visualize_tracking_results, 
                            compare_keypoints, visualize_reverse_validation_results, get_project_paths)
except ImportError:
    # When running as main, imports will be handled in main function
    pass


def regularize_image_and_keypoints(image: np.ndarray, 
                                 keypoints: Optional[List[Union[Dict, List]]] = None,
                                 max_image_size: int = 1024) -> Tuple[np.ndarray, Optional[List]]:
    """
    Standalone function to regularize image and keypoints for FlowFormer++ processing.
    
    Applies the following constraints:
    1. Maximum image dimension should not exceed max_image_size
    2. Both width and height should be divisible by 8
    3. Keypoint coordinates are scaled proportionally if image is resized
    
    Args:
        image: Input image as numpy array (H, W, 3)
        keypoints: Optional list of keypoints in either format:
                  - Dict format: [{'x': x, 'y': y}, ...]
                  - List format: [[x, y], ...]
        max_image_size: Maximum allowed dimension for regularization
        
    Returns:
        tuple: (processed_image, scaled_keypoints)
            - processed_image: Resized image meeting constraints
            - scaled_keypoints: Keypoints with coordinates adjusted for scaling (None if input was None)
            
    Note:
        Scale factors can be calculated as:
        original_h, original_w = image.shape[:2]
        processed_h, processed_w = processed_image.shape[:2]
        scale_w, scale_h = processed_w / original_w, processed_h / original_h
    """
    import cv2
    
    # Store original dimensions
    original_h, original_w = image.shape[:2]
    
    # Calculate target dimensions
    h, w = original_h, original_w
    
    # Scale down if image is too large
    if max(h, w) > max_image_size:
        scale = max_image_size / max(h, w)
        h, w = int(h * scale), int(w * scale)
    
    # Ensure dimensions are divisible by 8 (required by FlowFormer++)
    h = (h // 8) * 8
    w = (w // 8) * 8
    
    # Calculate scale factors for internal use
    scale_factors = (w / original_w, h / original_h)  # (scale_w, scale_h)
    
    # Resize image if necessary
    if (h, w) != (original_h, original_w):
        processed_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Scale keypoints if provided
        scaled_keypoints = None
        if keypoints is not None:
            scaled_keypoints = []
            for kp in keypoints:
                if isinstance(kp, dict):
                    # Dictionary format with 'x', 'y' keys
                    scaled_kp = kp.copy()
                    scaled_kp['x'] = kp['x'] * scale_factors[0]
                    scaled_kp['y'] = kp['y'] * scale_factors[1]
                    scaled_keypoints.append(scaled_kp)
                else:
                    # List format [x, y]
                    scaled_kp = [kp[0] * scale_factors[0], kp[1] * scale_factors[1]]
                    scaled_keypoints.append(scaled_kp)
    else:
        # No scaling needed
        processed_image = image.copy()
        scaled_keypoints = keypoints.copy() if keypoints is not None else None
    
    return processed_image, scaled_keypoints


class ReferenceImageData:
    """Container for reference image data with all associated metadata."""
    
    def __init__(self, 
                 original_image: np.ndarray, 
                 original_keypoints: Optional[List[Dict]] = None,
                 max_image_size: int = 1024):
        """Initialize reference image data with regularization.
        
        Args:
            original_image: Original input image (H, W, 3)
            original_keypoints: Original keypoints (list of dicts with 'x', 'y' keys)
            max_image_size: Maximum allowed dimension for regularization
        """
        # Store original data
        self.original_image = original_image.copy()
        self.original_keypoints = original_keypoints.copy() if original_keypoints else []
        self.original_size = original_image.shape[:2]  # (height, width)
        
        # Regularize for FlowFormer++ processing
        self._regularize_data(max_image_size)
    
    def _regularize_data(self, max_image_size: int):
        """Apply regularization constraints to image and keypoints using the standalone function."""
        (self.processed_image, 
         self.processed_keypoints) = regularize_image_and_keypoints(
             self.original_image, 
             self.original_keypoints, 
             max_image_size
         )
        
        # Calculate sizes and scale factors from the images directly
        self.processed_size = self.processed_image.shape[:2]  # (height, width)
        
        # Calculate scale factors: processed / original
        self.scale_factors = (
            self.processed_size[1] / self.original_size[1],  # width_scale
            self.processed_size[0] / self.original_size[0]   # height_scale
        )
    
    @property
    def was_regularized(self) -> bool:
        """Check if regularization was applied."""
        return self.processed_size != self.original_size
    
    @property
    def scale_factor(self) -> Tuple[float, float]:
        """Get scale factors as (width_scale, height_scale)."""
        return self.scale_factors
    
    @property
    def width_scale(self) -> float:
        """Get width scale factor."""
        return self.scale_factors[0]
    
    @property
    def height_scale(self) -> float:
        """Get height scale factor."""
        return self.scale_factors[1]
    
    def get_summary(self) -> Dict:
        """Get summary information about this reference image."""
        return {
            'original_shape': self.original_size + (3,),
            'processed_shape': self.processed_size + (3,),
            'scale_factors': {
                'width_scale': self.width_scale,
                'height_scale': self.height_scale
            },
            'keypoints_count': len(self.processed_keypoints),
            'regularization_applied': self.was_regularized
        }


class FFPPKeypointTracker:
    """FlowFormer++ based keypoint tracker with direct model loading.
    
    This class provides keypoint tracking functionality by loading the FlowFormer++
    model directly, eliminating the need for a separate server. It uses the same
    model loading approach as the FlowFormerPlusPlusServer.
    
    Public Interface:
    - set_reference_image(image, keypoints=None, image_name=None): Store reference image with keypoints
    - track_keypoints(target_image, reference_name=None, bidirectional=False): Track keypoints from stored reference to target image
    - remove_reference_image(image_name=None): Remove a stored reference image by name (None = default)
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 max_image_size: Optional[int] = None,
                 config: Optional[Dict] = None):
        """Initialize the FFPP keypoint tracker.
        
        Args:
            model_path: Path to FlowFormer++ checkpoint. If None, uses 'sintel.pth'.
            device: Device to run inference on ('cpu', 'cuda', 'auto').
            max_image_size: Maximum image size for processing. If None, uses config default.
            config: Configuration dictionary for processing parameters.
        """
        # Get project paths
        try:
            self.paths = get_project_paths()
        except NameError:
            # Fallback when running as main
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            self.paths = {
                'project_root': project_root,
                'thirdparty': os.path.join(project_root, 'ThirdParty')
            }
        self.config = self._load_config(config, max_image_size)
        
        # Model state
        self.model = None
        self.device = None
        self.model_loaded = False
        
        # Reference image state - support multiple reference images with keys
        self.reference_data = {}  # Dict[str, ReferenceImageData] - key -> reference data
        self.default_reference_key = None  # Key for default reference when None is passed
        
        # Processing state
        self.last_flow = None
        self.processing_stats = {}
        
        # Initialize model loading
        self._initialize_model(model_path, device)

    # ============================================================================
    # PUBLIC INTERFACE
    # ============================================================================

    def set_reference_image(self, 
                          image: np.ndarray, 
                          keypoints: Optional[List[Dict]] = None,
                          image_name: Optional[str] = None) -> Dict:
        """Set reference image for keypoint tracking with optional image key.
        
        This is the first of two main public methods. Use this to store reference 
        images with their associated keypoints for later tracking operations.
        
        Args:
            image: Reference image as numpy array (H, W, 3) in RGB format.
            keypoints: Optional list of keypoint dictionaries with 'x', 'y' keys.
            image_name: Optional string name to identify this reference image. 
                       If None, uses 'default' and sets as default reference.
            
        Returns:
            Dict with success status and information about the set reference image.
        """
        try:
            # Validate image
            if not isinstance(image, np.ndarray):
                return {
                    'success': False,
                    'error': f'Image must be numpy array, got {type(image)}',
                    'image_type': str(type(image))
                }
            
            if image.ndim != 3 or image.shape[2] != 3:
                return {
                    'success': False,
                    'error': f'Invalid image shape: {image.shape}. Expected (H, W, 3)',
                    'image_shape': image.shape
                }
            
            # Use 'default' key if none provided
            if image_name is None:
                image_name = 'default'
                self.default_reference_key = image_name
            
            # Handle keypoints - convert numpy array to list format if needed
            processed_keypoints = None
            if keypoints is not None:
                if isinstance(keypoints, np.ndarray):
                    # Convert numpy array to list of dict format
                    processed_keypoints = [{'x': float(kp[0]), 'y': float(kp[1])} for kp in keypoints]
                elif isinstance(keypoints, list):
                    processed_keypoints = keypoints
                else:
                    return {
                        'success': False,
                        'error': f'Keypoints must be list or numpy array, got {type(keypoints)}',
                        'keypoints_type': str(type(keypoints))
                    }
            
            # Create ReferenceImageData object (handles regularization automatically)
            ref_data = ReferenceImageData(
                original_image=image,
                original_keypoints=processed_keypoints,
                max_image_size=self.config.get('max_image_size', 1024)
            )
            
            # Store reference data
            self.reference_data[image_name] = ref_data
            
            # Set as default if it's the first reference or explicitly default
            if self.default_reference_key is None or image_name == 'default':
                self.default_reference_key = image_name
            
            # Return success information using the reference data
            summary = ref_data.get_summary()
            return {
                'success': True,
                'key': image_name,
                'original_image_shape': summary['original_shape'],
                'regularized_image_shape': summary['processed_shape'],
                'scale_factors': summary['scale_factors'],
                'keypoints_count': summary['keypoints_count'],
                'original_keypoints': ref_data.original_keypoints,
                'regularized_keypoints': ref_data.processed_keypoints,
                'is_default': image_name == self.default_reference_key,
                'total_reference_images': len(self.reference_data),
                'regularization_applied': summary['regularization_applied']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to set reference image: {str(e)}',
                'exception': str(e)
            }

    def track_keypoints(self, 
                       target_image: np.ndarray,
                       reference_name: Optional[str] = None,
                       bidirectional: bool = False) -> Dict:
        """Track keypoints from stored reference image to target image.
        
        This is the second of two main public methods. Use this to track keypoints
        from a stored reference image to a target image using optical flow.
        
        Args:
            target_image: Target image as numpy array (H, W, 3) in RGB format.
            reference_name: Name of stored reference image to use. If None, uses default reference.
            bidirectional: If True, compute reverse flow for accuracy validation. Tracks keypoints
                          forward (ref→target) then backward (target→ref) and measures consistency.
            
        Returns:
            dict: Tracking results with success status, tracked keypoints, and statistics.
                 The tracked keypoints are returned in the original target image coordinate system.
                 If bidirectional=True, includes accuracy metrics for each keypoint.
        
        Note:
            You must call set_reference_image() first to store a reference image with keypoints
            before using this function. The target image is automatically resized to match the
            reference image dimensions for flow computation, then keypoints are scaled back to
            the original target image coordinates.
            
            Bidirectional mode: Computes ref→target flow, then target→ref flow to validate
            tracking accuracy. Returns consistency distance for each keypoint.
        """
        try:
            # Determine which reference to use
            if reference_name is None:
                # Use default reference
                if self.default_reference_key is None or self.default_reference_key not in self.reference_data:
                    return {
                        'success': False,
                        'error': f'No default reference image available. Available references: {list(self.reference_data.keys())}. Use set_reference_image() first.'
                    }
                reference_name = self.default_reference_key
                image_source = 'default_stored'
            else:
                # Use specified reference
                if reference_name not in self.reference_data:
                    return {
                        'success': False,
                        'error': f'Reference image "{reference_name}" not found. Available references: {list(self.reference_data.keys())}'
                    }
                image_source = 'stored_by_name'
            
            # Get reference data
            ref_data = self.reference_data[reference_name]
            ref_img = ref_data.processed_image.copy()
            ref_keypoints = ref_data.processed_keypoints.copy()
            
            # Validate target image
            if not isinstance(target_image, np.ndarray):
                return {
                    'success': False,
                    'error': f'Target image must be numpy array, got {type(target_image)}'
                }
            
            if target_image.ndim != 3 or target_image.shape[2] != 3:
                return {
                    'success': False,
                    'error': f'Target image must be RGB format (H, W, 3), got {target_image.shape}'
                }
            
            # Resize target image to match reference dimensions exactly
            import cv2
            ref_height, ref_width = ref_img.shape[:2]
            orig_height, orig_width = target_image.shape[:2]
            target_img = cv2.resize(target_image, (ref_width, ref_height), interpolation=cv2.INTER_CUBIC)
            
            # Calculate scale factors to transform keypoints back to original target coordinates
            scale_x = orig_width / ref_width   # Scale factor for x coordinates
            scale_y = orig_height / ref_height  # Scale factor for y coordinates
            target_was_resized = (orig_height != ref_height) or (orig_width != ref_width)
            
            # Verify dimensions match
            if ref_img.shape != target_img.shape:
                return {
                    'success': False,
                    'error': f'Failed to resize target image: Reference is {ref_img.shape}, resized target is {target_img.shape}'
                }
            
            # Compute optical flow (forward: reference → target)
            start_time = time.time()
            forward_flow, forward_flow_stats = self._compute_flow(ref_img, target_img, return_stats=True)
            
            # Compute reverse flow if bidirectional validation is requested
            reverse_flow = None
            reverse_flow_stats = None
            if bidirectional:
                reverse_flow, reverse_flow_stats = self._compute_flow(target_img, ref_img, return_stats=True)
            
            # Track keypoints using forward flow with bilinear interpolation
            tracked_keypoints = []
            for i, kp in enumerate(ref_keypoints):
                x, y = kp['x'], kp['y']
                
                # Ensure coordinates are within flow bounds
                h, w = forward_flow.shape[:2]
                if x < 0 or x >= w or y < 0 or y >= h:
                    # Handle out-of-bounds points by clamping
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                
                # Get forward flow at keypoint location using bilinear interpolation
                dx_forward, dy_forward = self._bilinear_interpolate_flow(forward_flow, x, y)
                
                # Calculate new position in resized coordinate system
                new_x_resized = kp['x'] + dx_forward
                new_y_resized = kp['y'] + dy_forward
                
                # Bidirectional validation if enabled
                consistency_distance = None
                consistency_error_x = None
                consistency_error_y = None
                
                if bidirectional and reverse_flow is not None:
                    # Get reverse flow at the tracked position
                    # Clamp tracked position to reverse flow bounds
                    tracked_x_clamped = max(0, min(new_x_resized, w - 1))
                    tracked_y_clamped = max(0, min(new_y_resized, h - 1))
                    
                    dx_reverse, dy_reverse = self._bilinear_interpolate_flow(reverse_flow, tracked_x_clamped, tracked_y_clamped)
                    
                    # Apply reverse flow to get back to reference coordinates
                    back_x = new_x_resized + dx_reverse
                    back_y = new_y_resized + dy_reverse
                    
                    # Calculate consistency error (distance from original position)
                    consistency_error_x = back_x - kp['x']
                    consistency_error_y = back_y - kp['y']
                    consistency_distance = float((consistency_error_x**2 + consistency_error_y**2)**0.5)
                
                # Scale back to original target image coordinates
                if target_was_resized:
                    new_x_original = new_x_resized * scale_x
                    new_y_original = new_y_resized * scale_y
                else:
                    new_x_original = new_x_resized
                    new_y_original = new_y_resized
                
                # Create new keypoint by copying the original and updating x, y
                tracked_kp = kp.copy()  # Preserve all original keys and values
                tracked_kp['x'] = float(new_x_original)  # Update x coordinate (in original target scale)
                tracked_kp['y'] = float(new_y_original)  # Update y coordinate (in original target scale)
                
                # Add bidirectional validation metrics if enabled
                if bidirectional:
                    tracked_kp['consistency_distance'] = consistency_distance
                    tracked_kp['consistency_error_x'] = float(consistency_error_x) if consistency_error_x is not None else None
                    tracked_kp['consistency_error_y'] = float(consistency_error_y) if consistency_error_y is not None else None
                
                tracked_keypoints.append(tracked_kp)
            
            total_time = time.time() - start_time
            
            # Calculate bidirectional statistics if enabled
            bidirectional_stats = None
            if bidirectional and len(tracked_keypoints) > 0:
                consistency_distances = [kp.get('consistency_distance', 0) for kp in tracked_keypoints if kp.get('consistency_distance') is not None]
                if consistency_distances:
                    bidirectional_stats = {
                        'mean_consistency_distance': float(sum(consistency_distances) / len(consistency_distances)),
                        'max_consistency_distance': float(max(consistency_distances)),
                        'min_consistency_distance': float(min(consistency_distances)),
                        'consistent_keypoints': len(consistency_distances),
                        'total_keypoints': len(tracked_keypoints)
                    }
            
            return {
                'success': True,
                'tracked_keypoints': tracked_keypoints,
                'flow_computation_time': forward_flow_stats['processing_time'],
                'reverse_flow_computation_time': reverse_flow_stats['processing_time'] if reverse_flow_stats else None,
                'total_processing_time': total_time,
                'bidirectional_enabled': bidirectional,
                'bidirectional_stats': bidirectional_stats,
                'reference_name': reference_name,
                'reference_image_source': image_source,
                'reference_image_shape': ref_img.shape,
                'target_image_shape': target_img.shape,
                'original_target_shape': target_image.shape,
                'target_resized': target_was_resized,
                'target_scale_factors': {'x': scale_x, 'y': scale_y} if target_was_resized else None,
                'forward_flow_shape': forward_flow.shape,
                'reverse_flow_shape': reverse_flow.shape if reverse_flow is not None else None,
                'keypoints_count': len(tracked_keypoints),
                'reference_keypoints_count': len(ref_keypoints),
                'available_references': list(self.reference_data.keys()),
                'default_reference': self.default_reference_key
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Keypoint tracking failed: {str(e)}',
                'exception': str(e)
            }

    def remove_reference_image(self, image_name: Optional[str] = None) -> Dict:
        """Remove a stored reference image by name.
        
        Args:
            image_name: Name of the reference image to remove. If None, removes the default reference image.
            
        Returns:
            Dict with success status and information.
        """
        try:
            # Determine which key to remove
            key_to_remove = image_name
            if key_to_remove is None:
                # Use default reference key
                if self.default_reference_key is None:
                    return {
                        'success': False,
                        'error': 'No default reference image to remove. No reference images are currently stored.'
                    }
                key_to_remove = self.default_reference_key
            
            if key_to_remove not in self.reference_data:
                return {
                    'success': False,
                    'error': f'Reference image with key "{key_to_remove}" not found. Available keys: {list(self.reference_data.keys())}'
                }
            
            # Remove the reference data
            del self.reference_data[key_to_remove]
            
            # Update default key if necessary
            if self.default_reference_key == key_to_remove:
                if self.reference_data:
                    # Set first available as new default
                    self.default_reference_key = list(self.reference_data.keys())[0]
                else:
                    self.default_reference_key = None
            
            return {
                'success': True,
                'removed_key': key_to_remove,
                'input_key': image_name,  # Show what was actually passed in
                'new_default_key': self.default_reference_key,
                'remaining_keys': list(self.reference_data.keys()),
                'remaining_count': len(self.reference_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to remove reference image: {str(e)}',
                'exception': str(e)
            }

    # ============================================================================
    # PRIVATE METHODS
    # ============================================================================

    def _load_config(self, config: Optional[Dict], max_image_size: Optional[int] = None) -> Dict:
        """Load configuration with defaults."""
        default_config = {
            'max_image_size': 1024,
            'flow_computation': {
                'use_tiling': False,
                'use_cache': True
            },
            'preprocessing': {
                'normalize': True,
                'resize_mode': 'keep_aspect'
            }
        }
        
        if config:
            # Merge user config with defaults
            default_config.update(config)
        
        # Override max_image_size if explicitly provided
        if max_image_size is not None:
            default_config['max_image_size'] = max_image_size
        
        return default_config
    
    def _initialize_model(self, model_path: Optional[str], device: str):
        """Initialize FlowFormer++ model by loading it directly in the current environment."""
        try:
            # Use default model path if not provided
            if model_path is None:
                model_path = "checkpoints/sintel.pth"
            
            # Store model configuration
            self.model_path = model_path
            self.device_config = device
            
            # Add FlowFormer++ path and change working directory
            actual_ffp_path = os.path.join(self.paths['project_root'], 'ThirdParty', 'FlowFormerPlusPlusServer')
            sys.path.insert(0, actual_ffp_path)
            
            # Change to FlowFormer++ directory for relative imports
            original_cwd = os.getcwd()
            os.chdir(actual_ffp_path)
            
            try:
                # Temporarily remove our core module from sys.modules to avoid conflicts
                our_core_modules = {k: v for k, v in sys.modules.items() if k.startswith('core')}
                for k in our_core_modules:
                    del sys.modules[k]
                
                # Import and load model directly
                import visualize_flow_img_pair as flow_utils
                
                # Load model using the server's exact approach
                self.model, self.device = flow_utils.build_model(
                    model_path=model_path,
                    device_config=device,
                    use_cache=True
                )
                
                self.model_loaded = True
                
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
                # Restore our core modules
                for k, v in our_core_modules.items():
                    sys.modules[k] = v
                
        except Exception as e:
            print(f"❌ Failed to load FlowFormer++ model directly: {e}")
            print("💡 Make sure you're running in the 'flowformerpp' conda environment")
            raise
    
    def _compute_flow(self, 
                     image1: np.ndarray, 
                     image2: np.ndarray,
                     return_stats: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Compute optical flow between two images using directly loaded FlowFormer++ model.
        
        Args:
            image1: First image as numpy array (H, W, 3).
            image2: Second image as numpy array (H, W, 3).
            return_stats: Whether to return processing statistics.
            
        Returns:
            Flow array (H, W, 2) or tuple of (flow, stats) if return_stats=True.
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Make sure you're running in 'flowformerpp' conda environment.")
        
        start_time = time.time()
        
        try:
            # Use the server's exact fast function with our loaded model
            actual_ffp_path = os.path.join(self.paths['project_root'], 'ThirdParty', 'FlowFormerPlusPlusServer')
            original_cwd = os.getcwd()
            os.chdir(actual_ffp_path)
            
            try:
                # Temporarily remove our core module from sys.modules to avoid conflicts
                our_core_modules = {k: v for k, v in sys.modules.items() if k.startswith('core')}
                for k in our_core_modules:
                    del sys.modules[k]
                
                import visualize_flow_img_pair as flow_utils
                import torch
                
                # Convert numpy arrays directly to torch tensors (skip bytes conversion)
                image1_tensor = torch.from_numpy(image1.astype(np.uint8)).permute(2, 0, 1).float()
                image2_tensor = torch.from_numpy(image2.astype(np.uint8)).permute(2, 0, 1).float()
                
                # Call compute_flow_with_model directly with no_grad - much simpler!
                with torch.no_grad():
                    flow = flow_utils.compute_flow_with_model(
                        self.model, 
                        self.device, 
                        image1_tensor, 
                        image2_tensor, 
                        use_tiling=False
                    )
                
            finally:
                os.chdir(original_cwd)
                
                # Restore our core modules
                for k, v in our_core_modules.items():
                    sys.modules[k] = v
            
            # Store for potential reuse
            self.last_flow = flow
            
            # Calculate statistics
            processing_time = time.time() - start_time
            stats = {
                'processing_time': processing_time,
                'image_shape': image1.shape,
                'flow_shape': flow.shape,
                'using_direct_model': True,
                'optimization_method': 'direct_model_no_subprocess'
            }
            self.processing_stats = stats
            
            if return_stats:
                return flow, stats
            return flow
            
        except Exception as e:
            raise RuntimeError(f"Direct flow computation failed: {str(e)}")

    def _bilinear_interpolate_flow(self, flow: np.ndarray, x: float, y: float) -> tuple:
        """
        Bilinear interpolation to sample flow at sub-pixel locations.
        
        Args:
            flow: Flow field array (H, W, 2)
            x, y: Sub-pixel coordinates to sample at
            
        Returns:
            (dx, dy): Interpolated flow values
        """
        h, w = flow.shape[:2]
        
        # Clamp coordinates to valid range
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Get integer coordinates
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        # Get fractional parts
        fx, fy = x - x0, y - y0
        
        # Sample flow at four corners
        flow_00 = flow[y0, x0]  # top-left
        flow_10 = flow[y0, x1]  # top-right
        flow_01 = flow[y1, x0]  # bottom-left
        flow_11 = flow[y1, x1]  # bottom-right
        
        # Bilinear interpolation
        flow_top = flow_00 * (1 - fx) + flow_10 * fx
        flow_bottom = flow_01 * (1 - fx) + flow_11 * fx
        flow_interp = flow_top * (1 - fy) + flow_bottom * fy
        
        return float(flow_interp[0]), float(flow_interp[1])