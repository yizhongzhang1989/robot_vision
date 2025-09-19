"""
FlowFormer++ Keypoint Tracker Module
===================================

Direct FlowFormer++ model integration for keypoint tracking without cloud dependency.
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
        print("🚀 Initializing FlowFormer++ model...")
        
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
                print("✅ FlowFormer++ model loaded directly!")
                print(f"   Device: {self.device}")
                print("   Ready for ultra-fast flow computation!")
                
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


def test_simple():
    """Simple test: Initialize model and test basic keypoint tracking with sample data."""
    import cv2
    import json
    import numpy as np
    import time
    
    print("🧪 Test Simple - Basic Functionality")
    print("=" * 40)
    
    # ========================================
    # DATA PREPARATION PHASE
    # ========================================
    print("📁 Loading sample data...")
    ref_image_path = 'sample_data/flow_image_pair/ref_img.jpg'
    comp_image_path = 'sample_data/flow_image_pair/comp_img.jpg'
    ref_keypoints_path = 'sample_data/flow_image_pair/ref_img_keypoints.json'
    
    try:
        # Load images
        ref_img = cv2.imread(ref_image_path)
        target_img = cv2.imread(comp_image_path)
        
        if ref_img is None or target_img is None:
            print("❌ Failed to load sample images")
            return False
        
        # Load keypoints - use all keypoints for comprehensive test
        with open(ref_keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        
        # Use all original keypoints directly without modification
        test_keypoints = keypoints_data['keypoints']  # Use all keypoints as-is
        
        print(f"✅ Data loaded successfully:")
        print(f"   Reference image: {ref_img.shape}")
        print(f"   Target image: {target_img.shape}")  
        print(f"   Test keypoints: {len(test_keypoints)} (all keypoints)")
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return False
    
    # ========================================
    # FFPPKeypointTracker OPERATIONS
    # ========================================
    print("\n🚀 Initializing FFPPKeypointTracker...")
    init_start_time = time.time()
    tracker = FFPPKeypointTracker()
    init_elapsed_time = time.time() - init_start_time
    
    if not tracker.model_loaded:
        print("❌ Failed to load model")
        return False
    
    print(f"✅ FFPPKeypointTracker initialized on {tracker.device}")
    print(f"   Initialization time: {init_elapsed_time:.3f}s")
    
    # Test keypoint tracking with stored reference
    print("\n🎯 Testing tracker.track_keypoints() method...")
    print("   Mode: Using stored reference image")
    
    # First set up the reference image with keypoints
    ref_result = tracker.set_reference_image(ref_img, test_keypoints)
    if not ref_result['success']:
        print(f"❌ Failed to set reference image: {ref_result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Reference image set with {ref_result['keypoints_count']} keypoints")
    
    start_time = time.time()
    result = tracker.track_keypoints(target_img)
    elapsed_time = time.time() - start_time
    
    if result['success']:
        tracked_count = len(result.get('tracked_keypoints', []))
        print(f"✅ tracker.track_keypoints() successful!")
        print(f"   Time: {elapsed_time:.3f}s")
        print(f"   Tracked: {tracked_count} keypoints")
        print(f"   Processing time: {result.get('total_processing_time', 0):.3f}s")
        
        # Show some tracking results
        if tracked_count > 0:
            first_kp = result['tracked_keypoints'][0]
            print(f"   Example displacement: ({first_kp.get('displacement_x', 0):.1f}, {first_kp.get('displacement_y', 0):.1f})")
        
        # ========================================
        # VISUALIZATION AND OUTPUT
        # ========================================
        print("\n📊 Creating visualization...")
        try:
            import os
            
            # Create output directory if it doesn't exist
            output_dir = 'output'
            os.makedirs(output_dir, exist_ok=True)
            
            # Create visualization image
            vis_img = target_img.copy()
            
            # Draw tracked keypoints
            for i, kp in enumerate(result['tracked_keypoints']):
                x, y = int(round(kp['x'])), int(round(kp['y']))
                
                # Ensure coordinates are within image bounds
                h, w = vis_img.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    # Draw keypoint as circle
                    cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)  # Green filled circle
                    # Draw keypoint number
                    cv2.putText(vis_img, str(i+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Save visualization
            output_path = os.path.join(output_dir, 'test_simple_tracked_keypoints.jpg')
            cv2.imwrite(output_path, vis_img)
            print(f"✅ Visualization saved to: {output_path}")
            
            # Save tracking results as JSON
            json_output_path = os.path.join(output_dir, 'test_simple_results.json')
            output_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': 'simple_direct_tracking',
                'initialization_time': init_elapsed_time,
                'tracking_time': elapsed_time,
                'keypoints_count': tracked_count,
                'tracked_keypoints': result['tracked_keypoints'],
                'processing_stats': {
                    'flow_computation_time': result.get('flow_computation_time', 0),
                    'total_processing_time': result.get('total_processing_time', 0),
                    'reference_image_shape': result.get('reference_image_shape'),
                    'target_image_shape': result.get('target_image_shape')
                }
            }
            
            with open(json_output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Results saved to: {json_output_path}")
            
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
        
        return True
    else:
        print(f"❌ tracker.track_keypoints() failed: {result.get('error', 'Unknown error')}")
        return False


def test_ref_img():
    """Reference image test: Full functionality with file-based data."""
    import cv2
    import time
    import json
    import os
    import numpy as np
    
    print("\n🧪 Test Reference Images - Full Functionality")
    print("=" * 50)
    tracker = FFPPKeypointTracker()
    
    if not tracker.model_loaded:
        print("❌ Failed to load model. Exiting...")
        return
    
    print(f"✅ Model loaded on {tracker.device}")
    
    # Load sample images and keypoints
    print("\n📁 Loading sample data...")
    ref_image_path = 'sample_data/flow_image_pair/ref_img.jpg'
    comp_image_path = 'sample_data/flow_image_pair/comp_img.jpg'
    ref_keypoints_path = 'sample_data/flow_image_pair/ref_img_keypoints.json'
    
    try:
        # Load reference image
        ref_image = cv2.imread(ref_image_path)
        if ref_image is None:
            raise FileNotFoundError(f"Could not load reference image: {ref_image_path}")
        
        # Load comparison image
        comp_image = cv2.imread(comp_image_path)
        if comp_image is None:
            raise FileNotFoundError(f"Could not load comparison image: {comp_image_path}")
        
        # Load keypoints
        with open(ref_keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        
        ref_keypoints = np.array([[int(kp['x']), int(kp['y'])] for kp in keypoints_data['keypoints']])
        print(f"📍 Loaded {len(ref_keypoints)} keypoints")
        
        # Create a second reference image (duplicate for demo)
        ref_image_2 = ref_image.copy()
        ref_keypoints_2 = ref_keypoints + 10  # Slightly offset keypoints
        
    except Exception as e:
        print(f"❌ Error loading sample data: {str(e)}")
        return
    
    # Set up multiple reference images
    print("\n📋 Setting up references...")
    
    tracker.set_reference_image(ref_image, ref_keypoints)
    tracker.set_reference_image(ref_image_2, ref_keypoints_2, image_name="ref_offset")
    tracker.set_reference_image(ref_image, image_name="ref_image_only")
    
    print(f"   References: {list(tracker.reference_data.keys())}")
    print(f"   Default: {tracker.default_reference_key}")
    
    # Show tracking results
    print("\n📁 Reference images information:")
    print(f"📍 Reference images set: Primary key='{tracker.default_reference_key}' and secondary key='ref_offset'")
    
    # Demonstrate different tracking modes
    print("\n🎯 Testing tracking...")
    
    # Test default reference
    start_time = time.time()
    result_default = tracker.track_keypoints(comp_image)
    elapsed_time = time.time() - start_time
    print(f"   Default: {elapsed_time:.3f}s - {len(result_default.get('tracked_keypoints', []))} points")
    
    # Test specific reference by name
    start_time = time.time()
    result_key = tracker.track_keypoints(comp_image, reference_name="ref_offset")
    elapsed_time = time.time() - start_time
    print(f"   By name: {elapsed_time:.3f}s - {len(result_key.get('tracked_keypoints', []))} points")
    
    # Test third reference (ref_image_only has no keypoints, should show 0 points)
    start_time = time.time()
    result_no_keypoints = tracker.track_keypoints(comp_image, reference_name="ref_image_only")
    elapsed_time = time.time() - start_time
    print(f"   No keypoints: {elapsed_time:.3f}s - {len(result_no_keypoints.get('tracked_keypoints', []))} points")
    
    # Test removal functionality
    print("\n🗑️ Testing removal...")
    print(f"   Before: {list(tracker.reference_data.keys())}")
    tracker.remove_reference_image("ref_image_only")
    tracker.remove_reference_image(None)  # Remove default
    print(f"   After: {list(tracker.reference_data.keys())}")
    
    # Save results
    if result_default['success']:
        print("\n💾 Saving results...")
        output_path = 'output/tracked_keypoints_demo.json'
        
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tracking_results': {
                'default_mode': result_default.get('tracked_keypoints', []),
                'by_name_mode': result_key.get('tracked_keypoints', []),
                'no_keypoints_mode': result_no_keypoints.get('tracked_keypoints', [])
            }
        }
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"   Saved to: {output_path}")
    
    print(f"\n✅ Demo completed - {len(tracker.reference_data)} references remaining")
    print(f"   Device: {tracker.device} | Default: {tracker.default_reference_key}")
    
    return True


def main():
    """Main function - runs all tests in sequence."""
    print("🔥 FFpp Keypoint Tracker Test Suite 🔥")
    print("=" * 45)
    
    # Run test sequence
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Simple functionality
    try:
        if test_simple():
            tests_passed += 1
            print("✅ test_simple PASSED")
        else:
            print("❌ test_simple FAILED")
    except Exception as e:
        print(f"❌ test_simple ERROR: {e}")
    
    # Test 2: Reference image functionality  
    try:
        if test_ref_img():
            tests_passed += 1
            print("✅ test_ref_img PASSED")
        else:
            print("❌ test_ref_img FAILED")
    except Exception as e:
        print(f"❌ test_ref_img ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 45)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    if tests_passed == total_tests:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️  Some tests failed - check output above")


if __name__ == "__main__":
    main()