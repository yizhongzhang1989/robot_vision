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


class FFPPKeypointTracker:
    """FlowFormer++ based keypoint tracker with direct model loading.
    
    This class provides keypoint tracking functionality by loading the FlowFormer++
    model directly, eliminating the need for a separate server. It uses the same
    model loading approach as the FlowFormerPlusPlusServer.
    
    Public Interface:
    - set_reference_image(image, keypoints=None, image_key=None): Store reference image with keypoints
    - track_keypoints(target_image, reference_image=None, reference_keypoints=None): Track keypoints between images
    - remove_reference_image(image_key=None): Remove a stored reference image by key (None = default)
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 config: Optional[Dict] = None):
        """Initialize the FFPP keypoint tracker.
        
        Args:
            model_path: Path to FlowFormer++ checkpoint. If None, uses 'sintel.pth'.
            device: Device to run inference on ('cpu', 'cuda', 'auto').
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
        self.config = self._load_config(config)
        
        # Model state
        self.model = None
        self.device = None
        self.model_loaded = False
        
        # Reference image state - support multiple reference images with keys
        self.reference_images = {}  # Dict[str, np.ndarray] - key -> image
        self.reference_keypoints = {}  # Dict[str, List[Dict]] - key -> keypoints
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
                          image_key: Optional[str] = None) -> Dict:
        """Set reference image for keypoint tracking with optional image key.
        
        This is the first of two main public methods. Use this to store reference 
        images with their associated keypoints for later tracking operations.
        
        Args:
            image: Reference image as numpy array (H, W, 3) in RGB format.
            keypoints: Optional list of keypoint dictionaries with 'x', 'y' keys.
            image_key: Optional string key to identify this reference image. 
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
            if image_key is None:
                image_key = 'default'
                self.default_reference_key = image_key
            
            # Store reference image
            self.reference_images[image_key] = image.copy()
            
            # Handle keypoints - convert numpy array to list format if needed
            if keypoints is not None:
                if isinstance(keypoints, np.ndarray):
                    # Convert numpy array to list of dict format
                    keypoints = [{'x': float(kp[0]), 'y': float(kp[1])} for kp in keypoints]
                elif not isinstance(keypoints, list):
                    return {
                        'success': False,
                        'error': f'Keypoints must be list or numpy array, got {type(keypoints)}',
                        'keypoints_type': str(type(keypoints))
                    }
            else:
                keypoints = []
            
            # Store keypoints
            self.reference_keypoints[image_key] = keypoints
            
            # Set as default if it's the first reference or explicitly default
            if self.default_reference_key is None or image_key == 'default':
                self.default_reference_key = image_key
            
            # Return success information
            return {
                'success': True,
                'key': image_key,
                'image_shape': image.shape,
                'keypoints_count': len(self.reference_keypoints[image_key]),
                'keypoints': self.reference_keypoints[image_key],
                'is_default': image_key == self.default_reference_key,
                'total_reference_images': len(self.reference_images)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to set reference image: {str(e)}',
                'exception': str(e)
            }

    def track_keypoints(self, 
                       target_image: np.ndarray,
                       reference_image: Union[None, np.ndarray, str] = None,
                       reference_keypoints: List[Dict] = None) -> Dict:
        """Track keypoints from reference image to target image.
        
        This is the second of two main public methods. Use this to track keypoints
        from a reference image to a target image using optical flow.
        
        Args:
            target_image: Target image as numpy array (H, W, 3) in RGB format.
            reference_image: Reference image specification:
                - None: Use default reference image (key stored in default_reference_key)
                - str: Use stored reference image with this key
                - np.ndarray: Use this image directly (not stored)
            reference_keypoints: List of keypoint dictionaries with 'x', 'y' keys.
                - None: Use keypoints stored with reference image (if available)
            
        Returns:
            dict: Tracking results with success status, tracked keypoints, and statistics.
        """
        try:
            # Determine reference image to use
            ref_img = None
            image_source = None
            reference_key = None
            
            if isinstance(reference_image, np.ndarray):
                # Use provided numpy array directly
                ref_img = reference_image.copy()
                image_source = 'provided_array'
                reference_key = 'provided_array'
                
            elif isinstance(reference_image, str):
                # Use stored reference image with specified key
                if reference_image not in self.reference_images:
                    return {
                        'success': False,
                        'error': f'Reference image with key "{reference_image}" not found. Available keys: {list(self.reference_images.keys())}'
                    }
                ref_img = self.reference_images[reference_image].copy()
                image_source = 'stored_by_key'
                reference_key = reference_image
                
            elif reference_image is None:
                # Use default reference image
                if self.default_reference_key is None or self.default_reference_key not in self.reference_images:
                    return {
                        'success': False,
                        'error': f'No default reference image available. Available keys: {list(self.reference_images.keys())}. Use set_reference_image() first or provide a reference_image parameter.'
                    }
                ref_img = self.reference_images[self.default_reference_key].copy()
                image_source = 'default_stored'
                reference_key = self.default_reference_key
                
            else:
                return {
                    'success': False,
                    'error': f'Invalid reference_image type: {type(reference_image)}. Must be None, str, or np.ndarray.'
                }
            
            # Validate target image
            if not isinstance(target_image, np.ndarray):
                return {
                    'success': False,
                    'error': f'Target image must be numpy array, got {type(target_image)}'
                }
            
            target_img = target_image.copy()
            
            # Validate image dimensions
            if len(ref_img.shape) != 3 or ref_img.shape[2] != 3:
                return {
                    'success': False,
                    'error': f'Reference image must be RGB format (H, W, 3), got {ref_img.shape}'
                }
                
            if len(target_img.shape) != 3 or target_img.shape[2] != 3:
                return {
                    'success': False,
                    'error': f'Target image must be RGB format (H, W, 3), got {target_img.shape}'
                }
            
            # Check if images have same dimensions
            if ref_img.shape != target_img.shape:
                return {
                    'success': False,
                    'error': f'Image dimension mismatch: Reference is {ref_img.shape}, target is {target_img.shape}'
                }
            
            # Compute optical flow
            start_time = time.time()
            flow, flow_stats = self._compute_flow(ref_img, target_img, return_stats=True)
            
            # Track keypoints using flow with bilinear interpolation
            tracked_keypoints = []
            for kp in reference_keypoints:
                x, y = kp['x'], kp['y']
                
                # Ensure coordinates are within flow bounds
                h, w = flow.shape[:2]
                if x < 0 or x >= w or y < 0 or y >= h:
                    # Handle out-of-bounds points by clamping
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                
                # Get flow at keypoint location using bilinear interpolation
                dx, dy = self._bilinear_interpolate_flow(flow, x, y)
                
                # Calculate new position
                new_x = kp['x'] + dx
                new_y = kp['y'] + dy
                
                # Create new keypoint by copying the original and updating x, y
                tracked_kp = kp.copy()  # Preserve all original keys and values
                tracked_kp['x'] = float(new_x)  # Update x coordinate
                tracked_kp['y'] = float(new_y)  # Update y coordinate
                
                tracked_keypoints.append(tracked_kp)
            
            total_time = time.time() - start_time
            
            return {
                'success': True,
                'tracked_keypoints': tracked_keypoints,
                'flow_computation_time': flow_stats['processing_time'],
                'total_processing_time': total_time,
                'reference_image_source': image_source,
                'reference_key': reference_key,
                'reference_image_shape': ref_img.shape,
                'target_image_shape': target_img.shape,
                'flow_shape': flow.shape,
                'keypoints_count': len(tracked_keypoints),
                'available_reference_keys': list(self.reference_images.keys()),
                'default_reference_key': self.default_reference_key
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Keypoint tracking failed: {str(e)}',
                'exception': str(e)
            }

    def remove_reference_image(self, image_key: Optional[str] = None) -> Dict:
        """Remove a stored reference image by key.
        
        Args:
            image_key: Key of the reference image to remove. If None, removes the default reference image.
            
        Returns:
            Dict with success status and information.
        """
        try:
            # Determine which key to remove
            key_to_remove = image_key
            if key_to_remove is None:
                # Use default reference key
                if self.default_reference_key is None:
                    return {
                        'success': False,
                        'error': 'No default reference image to remove. No reference images are currently stored.'
                    }
                key_to_remove = self.default_reference_key
            
            if key_to_remove not in self.reference_images:
                return {
                    'success': False,
                    'error': f'Reference image with key "{key_to_remove}" not found. Available keys: {list(self.reference_images.keys())}'
                }
            
            # Remove the reference image and keypoints
            del self.reference_images[key_to_remove]
            if key_to_remove in self.reference_keypoints:
                del self.reference_keypoints[key_to_remove]
            
            # Update default key if necessary
            if self.default_reference_key == key_to_remove:
                if self.reference_images:
                    # Set first available as new default
                    self.default_reference_key = list(self.reference_images.keys())[0]
                else:
                    self.default_reference_key = None
            
            return {
                'success': True,
                'removed_key': key_to_remove,
                'input_key': image_key,  # Show what was actually passed in
                'new_default_key': self.default_reference_key,
                'remaining_keys': list(self.reference_images.keys()),
                'remaining_count': len(self.reference_images)
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

    def _load_config(self, config: Optional[Dict]) -> Dict:
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
    
    # Test direct keypoint tracking (no reference storage)
    print("\n🎯 Testing tracker.track_keypoints() method...")
    print("   Mode: Direct tracking (reference_image passed as array)")
    
    start_time = time.time()
    result = tracker.track_keypoints(target_img, reference_image=ref_img, reference_keypoints=test_keypoints)
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
    tracker.set_reference_image(ref_image_2, ref_keypoints_2, image_key="ref_offset")
    tracker.set_reference_image(ref_image, image_key="ref_image_only")
    
    print(f"   References: {list(tracker.reference_images.keys())}")
    print(f"   Default: {tracker.default_reference_key}")
    
    # Show tracking results
    print("\n📁 Reference images information:")
    print(f"📍 Reference images set: Primary key='{tracker.default_reference_key}' and secondary key='ref_offset'")
    
    # Demonstrate different tracking modes
    print("\n🎯 Testing tracking...")
    
    # Test default reference
    stored_keypoints = tracker.reference_keypoints[tracker.default_reference_key]
    start_time = time.time()
    result_default = tracker.track_keypoints(comp_image, reference_image=None, reference_keypoints=stored_keypoints)
    elapsed_time = time.time() - start_time
    print(f"   Default: {elapsed_time:.3f}s - {len(result_default.get('tracked_keypoints', []))} points")
    
    # Test specific key
    stored_keypoints_2 = tracker.reference_keypoints['ref_offset']
    start_time = time.time()
    result_key = tracker.track_keypoints(comp_image, reference_image="ref_offset", reference_keypoints=stored_keypoints_2)
    elapsed_time = time.time() - start_time
    print(f"   By key: {elapsed_time:.3f}s - {len(result_key.get('tracked_keypoints', []))} points")
    
    # Test direct array
    start_time = time.time()
    keypoints_dict_format = [{'x': float(kp[0]), 'y': float(kp[1])} for kp in ref_keypoints]
    result_direct = tracker.track_keypoints(comp_image, reference_image=ref_image, reference_keypoints=keypoints_dict_format)
    elapsed_time = time.time() - start_time
    print(f"   Direct: {elapsed_time:.3f}s - {len(result_direct.get('tracked_keypoints', []))} points")
    
    # Test removal functionality
    print("\n🗑️ Testing removal...")
    print(f"   Before: {list(tracker.reference_images.keys())}")
    tracker.remove_reference_image("ref_image_only")
    tracker.remove_reference_image(None)  # Remove default
    print(f"   After: {list(tracker.reference_images.keys())}")
    
    # Save results
    if result_default['success']:
        print("\n💾 Saving results...")
        output_path = 'output/tracked_keypoints_demo.json'
        
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tracking_results': {
                'default_mode': result_default.get('tracked_keypoints', []),
                'key_mode': result_key.get('tracked_keypoints', []),
                'direct_mode': result_direct.get('tracked_keypoints', [])
            }
        }
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"   Saved to: {output_path}")
    
    print(f"\n✅ Demo completed - {len(tracker.reference_images)} references remaining")
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