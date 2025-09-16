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
        
        # Processing state
        self.last_flow = None
        self.processing_stats = {}
        
        # Initialize model loading
        self._initialize_model(model_path, device)
    
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
            ffp_path = os.path.join(self.paths['thirdparty'], 'FlowFormerPlusPlusServer')
            print(f"DEBUG: FlowFormer++ path: {ffp_path}")
            print(f"DEBUG: Path exists: {os.path.exists(ffp_path)}")
            
            # Actually use the correct path - just the FlowFormerPlusPlusServer directory
            actual_ffp_path = os.path.join(self.paths['project_root'], 'ThirdParty', 'FlowFormerPlusPlusServer')
            print(f"DEBUG: Actual FlowFormer++ path: {actual_ffp_path}")
            print(f"DEBUG: Actual path exists: {os.path.exists(actual_ffp_path)}")
            
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
    
    def compute_flow(self, 
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
            # Convert images to bytes format for server's compute_flow_from_bytes function
            from io import BytesIO
            from PIL import Image
            
            # Convert numpy arrays to PIL Images
            img1_pil = Image.fromarray(image1.astype(np.uint8))
            img2_pil = Image.fromarray(image2.astype(np.uint8))
            
            # Convert to bytes
            img1_bytes = BytesIO()
            img2_bytes = BytesIO()
            img1_pil.save(img1_bytes, format='PNG')
            img2_pil.save(img2_bytes, format='PNG')
            img1_bytes = img1_bytes.getvalue()
            img2_bytes = img2_bytes.getvalue()
            
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
                
                # Use the ultra-fast cached computation - this should be ~0.3s!
                flow = flow_utils.compute_flow_from_bytes(
                    img1_bytes, img2_bytes,
                    model_path=None,  # Use our pre-loaded model
                    device_config=str(self.device),
                    max_size=400,
                    use_cache=True
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
    
    def track_keypoints_with_flow(self, keypoints: List[Dict], flow: np.ndarray) -> List[Dict]:
        """Track keypoints using precomputed optical flow.
        
        Args:
            keypoints: List of keypoint dictionaries with 'x', 'y', 'name' keys.
            flow: Optical flow array (H, W, 2).
            
        Returns:
            List of tracked keypoint dictionaries.
        """
        tracked_keypoints = []
        
        for kp in keypoints:
            x, y = int(kp['x']), int(kp['y'])
            
            # Ensure coordinates are within flow bounds
            h, w = flow.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            
            # Get flow at keypoint location
            dx, dy = flow[y, x]
            
            # Calculate new position
            new_x = x + dx
            new_y = y + dy
            
            tracked_kp = {
                'name': kp['name'],
                'x': float(new_x),
                'y': float(new_y),
                'original_x': float(kp['x']),
                'original_y': float(kp['y']),
                'displacement': float(np.sqrt(dx**2 + dy**2))
            }
            
            tracked_keypoints.append(tracked_kp)
        
        return tracked_keypoints
    
    def track_keypoints(self, 
                       keypoints: List[Dict], 
                       image1: np.ndarray, 
                       image2: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Track keypoints between two images.
        
        Args:
            keypoints: List of keypoint dictionaries.
            image1: First image (reference).
            image2: Second image (comparison).
            
        Returns:
            Tuple of (tracked_keypoints, flow).
        """
        # Compute optical flow
        flow = self.compute_flow(image1, image2)
        
        # Track keypoints using flow
        tracked_keypoints = self.track_keypoints_with_flow(keypoints, flow)
        
        return tracked_keypoints, flow
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_loaded': self.model_loaded,
            'device': str(self.device),
            'config': self.config,
            'last_processing_stats': self.processing_stats
        }
    
    def clear_cache(self):
        """Clear model caches to free memory."""
        # Note: Model caching is handled by the subprocess approach
        self.last_flow = None
        self.processing_stats = {}
    
    def __del__(self):
        """Cleanup resources."""
        self.clear_cache()


def main():
    """Test function to demonstrate FlowFormer++ keypoint tracking with timing."""
    import time
    import json
    import sys
    import os
    from PIL import Image
    
    # Add project root to Python path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    print("🧪 FlowFormer++ Keypoint Tracker Test")
    print("=" * 50)
    
    # Test data paths
    ref_img_path = "sample_data/flow_image_pair/ref_img.jpg"
    comp_img_path = "sample_data/flow_image_pair/comp_img.jpg"
    keypoints_path = "sample_data/flow_image_pair/ref_img_keypoints.json"
    
    try:
        # Step 1: Initialize and load model
        print("\n📋 Step 1: Model Initialization")
        print("-" * 30)
        model_start_time = time.time()
        
        tracker = FFPPKeypointTracker()
        # Model loads automatically during initialization
        
        model_load_time = time.time() - model_start_time
        print(f"✅ Model loaded in {model_load_time:.2f} seconds")
        
        # Load test data
        print("\n📋 Step 2: Loading Test Data")
        print("-" * 30)
        data_start_time = time.time()
        
        # Load images
        ref_img = np.array(Image.open(ref_img_path))
        comp_img = np.array(Image.open(comp_img_path))
        
        # Load keypoints
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        keypoints = keypoints_data.get('keypoints', [])
        
        data_load_time = time.time() - data_start_time
        print(f"✅ Test data loaded in {data_load_time:.3f} seconds")
        print(f"   - Reference image: {ref_img.shape}")
        print(f"   - Comparison image: {comp_img.shape}")
        print(f"   - Keypoints: {len(keypoints)}")
        
        # Step 3: Iterate tracking 5 times to test performance
        print("\n📋 Step 3: Iterative Keypoint Tracking (5 iterations)")
        print("-" * 50)
        
        iteration_times = []
        all_tracked_keypoints = []
        
        for iteration in range(1, 6):
            print(f"\n🔄 Iteration {iteration}/5:")
            iter_start_time = time.time()
            
            tracked_keypoints, flow = tracker.track_keypoints(keypoints, ref_img, comp_img)
            
            iter_time = time.time() - iter_start_time
            iteration_times.append(iter_time)
            all_tracked_keypoints.append(tracked_keypoints)
            
            print(f"   ✅ Completed in {iter_time:.2f} seconds")
            print(f"   📊 Flow shape: {flow.shape}")
            print(f"   🎯 Tracked {len(tracked_keypoints)} keypoints")
        
        # Calculate statistics
        avg_time = sum(iteration_times) / len(iteration_times)
        min_time = min(iteration_times)
        max_time = max(iteration_times)
        
        total_tracking_time = sum(iteration_times)
        total_time = time.time() - model_start_time
        
        # Summary
        print("\n📊 Performance Summary")
        print("=" * 50)
        print(f"Model initialization:     {model_load_time:.2f}s")
        print(f"Data loading:             {data_load_time:.3f}s")
        print(f"Total tracking time:      {total_tracking_time:.2f}s (5 iterations)")
        print("-" * 30)
        print(f"Per-iteration timing:")
        for i, iter_time in enumerate(iteration_times, 1):
            print(f"  Iteration {i}:           {iter_time:.2f}s")
        print("-" * 30)
        print(f"Average iteration time:   {avg_time:.2f}s")
        print(f"Fastest iteration:        {min_time:.2f}s")
        print(f"Slowest iteration:        {max_time:.2f}s")
        print(f"Performance improvement:  {((max_time - min_time) / max_time * 100):.1f}%")
        print("-" * 30)
        print(f"Total test time:          {total_time:.2f}s")
        
        # Show some tracking results from the last iteration
        print("\n🎯 Sample Tracking Results (Last Iteration)")
        print("-" * 40)
        final_tracked_keypoints = all_tracked_keypoints[-1]
        for i, kp in enumerate(final_tracked_keypoints[:5]):  # Show first 5
            displacement = kp['displacement']
            print(f"Point {i+1}: moved {displacement:.1f} pixels")
        if len(final_tracked_keypoints) > 5:
            print(f"... and {len(final_tracked_keypoints) - 5} more points")
            
        print("\n🎉 Test completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()