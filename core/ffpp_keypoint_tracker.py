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
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image

from core.utils import (load_keypoints, resize_keypoints, visualize_tracking_results, 
                        compare_keypoints, visualize_reverse_validation_results, get_project_paths)


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
        self.paths = get_project_paths()
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
        """Initialize and load the FlowFormer++ model using the server's approach."""
        print("🚀 Initializing FlowFormer++ model...")
        
        try:
            # Add FlowFormer++ path to sys.path
            ffp_path = os.path.join(self.paths['thirdparty'], 'FlowFormerPlusPlusServer')
            if ffp_path not in sys.path:
                sys.path.insert(0, ffp_path)
            
            # Import the build_model function from the server code
            import visualize_flow_img_pair as flow_utils
            
            # Use default model path if not provided
            if model_path is None:
                model_path = os.path.join(ffp_path, 'checkpoints', 'sintel.pth')
            elif not os.path.isabs(model_path):
                # If relative path, make it relative to FlowFormerPlusPlusServer
                model_path = os.path.join(ffp_path, model_path)
            
            # Build model using the same function as the web server
            self.model, self.device = flow_utils.build_model(
                model_path=model_path,
                device_config=device,
                use_cache=self.config['flow_computation']['use_cache']
            )
            
            self.model_loaded = True
            print("✅ FlowFormer++ model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load FlowFormer++ model: {e}")
            raise
    
    def compute_flow(self, 
                     image1: np.ndarray, 
                     image2: np.ndarray,
                     return_stats: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Compute optical flow between two images.
        
        Args:
            image1: First image as numpy array (H, W, 3).
            image2: Second image as numpy array (H, W, 3).
            return_stats: Whether to return processing statistics.
            
        Returns:
            Flow array (H, W, 2) or tuple of (flow, stats) if return_stats=True.
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call _initialize_model() first.")
        
        start_time = time.time()
        
        # Convert images to PIL format, then to bytes for compatibility with server code
        from io import BytesIO
        img1_pil = Image.fromarray(image1.astype(np.uint8))
        img2_pil = Image.fromarray(image2.astype(np.uint8))
        
        img1_bytes = BytesIO()
        img2_bytes = BytesIO()
        img1_pil.save(img1_bytes, format='PNG')
        img2_pil.save(img2_bytes, format='PNG')
        img1_bytes = img1_bytes.getvalue()
        img2_bytes = img2_bytes.getvalue()
        
        # Add FlowFormer++ path to sys.path if not already added
        ffp_path = os.path.join(self.paths['thirdparty'], 'FlowFormerPlusPlusServer')
        if ffp_path not in sys.path:
            sys.path.insert(0, ffp_path)
        
        # Use the server's flow computation function
        import visualize_flow_img_pair as flow_utils
        flow = flow_utils.compute_flow_with_model(
            self.model, 
            self.device,
            self._prepare_image_tensor(image1),
            self._prepare_image_tensor(image2),
            use_tiling=self.config['flow_computation']['use_tiling']
        )
        
        # Store for potential reuse
        self.last_flow = flow
        
        # Calculate statistics
        processing_time = time.time() - start_time
        stats = {
            'processing_time': processing_time,
            'image_shape': image1.shape,
            'flow_shape': flow.shape,
            'device': str(self.device),
            'memory_used': self._get_memory_usage()
        }
        self.processing_stats = stats
        
        if return_stats:
            return flow, stats
        return flow
    
    def _prepare_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for model input."""
        # Convert to RGB if BGR
        if image.shape[-1] == 3:
            image = image[..., ::-1]  # BGR to RGB
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        if self.config['preprocessing']['normalize']:
            image_tensor = image_tensor / 255.0
        
        return image_tensor
    
    def _get_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        stats = {}
        
        if self.device.type == 'cuda':
            stats['gpu_allocated'] = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            stats['gpu_reserved'] = torch.cuda.memory_reserved(self.device) / 1024**2   # MB
        
        return stats
    
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
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        self.last_flow = None
        self.processing_stats = {}
    
    def __del__(self):
        """Cleanup resources."""
        self.clear_cache()