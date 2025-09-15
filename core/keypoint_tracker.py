"""
Keypoint Tracker Module
=======================

Main keypoint tracking functionality using optical flow.
"""

import os
import json
import time
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

from core.utils import (load_keypoints, resize_keypoints, visualize_tracking_results, 
                        compare_keypoints, visualize_reverse_validation_results, get_project_paths)

# Add ThirdParty to path for flowformer_api import
_paths = get_project_paths()
sys.path.insert(0, _paths['thirdparty'])

try:
    from flowformer_api import FlowFormerClient
except ImportError:
    FlowFormerClient = None


class KeypointTracker:
    """Keypoint tracking using optical flow."""
    
    def __init__(self, server_url="http://msraig-ubuntu-3:5000"):
        """Initialize the keypoint tracker.
        
        Args:
            server_url: URL of the FlowFormer server
        """
        self.server_url = server_url
        self.paths = get_project_paths()
        
        if FlowFormerClient is None:
            raise ImportError("FlowFormer API not found. Check ThirdParty directory.")
        
        self.client = FlowFormerClient(server_url=server_url)
    
    def track_keypoints_with_flow(self, keypoints, flow):
        """Track keypoints using optical flow."""
        tracked_keypoints = []
        
        for kp in keypoints:
            # Get integer coordinates for flow lookup
            x = int(round(kp['x']))
            y = int(round(kp['y']))
            
            # Ensure coordinates are within flow bounds
            h, w = flow.shape[:2]
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            
            # Get flow vector at keypoint location
            flow_x = flow[y, x, 0]
            flow_y = flow[y, x, 1]
            
            # Calculate new position
            new_x = kp['x'] + flow_x
            new_y = kp['y'] + flow_y
            
            tracked_kp = kp.copy()
            tracked_kp['new_x'] = new_x
            tracked_kp['new_y'] = new_y
            tracked_kp['flow_x'] = flow_x
            tracked_kp['flow_y'] = flow_y
            tracked_keypoints.append(tracked_kp)
        
        return tracked_keypoints
    
    def load_images(self, ref_img_path=None, comp_img_path=None, target_width=800):
        """Load reference and comparison images.
        
        Args:
            ref_img_path: Path to reference image
            comp_img_path: Path to comparison image
            target_width: Target width for resizing (None to keep original size)
        """
        if ref_img_path is None:
            ref_img_path = os.path.join(self.paths['sample_data'], 'flow_image_pair', 'ref_img.jpg')
        if comp_img_path is None:
            comp_img_path = os.path.join(self.paths['sample_data'], 'flow_image_pair', 'comp_img.jpg')
            
        # Load images as PIL first for resizing
        ref_img_pil = Image.open(ref_img_path)
        comp_img_pil = Image.open(comp_img_path)
        
        # Resize if target_width is specified
        if target_width is not None:
            # Resize reference image
            orig_w1, orig_h1 = ref_img_pil.size
            aspect_ratio1 = orig_h1 / orig_w1
            target_height1 = int(target_width * aspect_ratio1)
            ref_img_pil = ref_img_pil.resize((target_width, target_height1), Image.Resampling.LANCZOS)
            
            # Resize comparison image
            orig_w2, orig_h2 = comp_img_pil.size
            aspect_ratio2 = orig_h2 / orig_w2
            target_height2 = int(target_width * aspect_ratio2)
            comp_img_pil = comp_img_pil.resize((target_width, target_height2), Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        ref_img = np.array(ref_img_pil)
        comp_img = np.array(comp_img_pil)
        
        return ref_img, comp_img
    
    def run_tracking(self, keypoints_json_path=None, ref_img_path=None, comp_img_path=None, 
                     output_dir=None, verbose=True, target_width=800):
        """Run complete keypoint tracking workflow.
        
        Args:
            keypoints_json_path: Path to keypoints JSON file
            ref_img_path: Path to reference image  
            comp_img_path: Path to comparison image
            output_dir: Output directory for results
            verbose: Whether to print progress messages
            target_width: Target width for image resizing (None to keep original size)
            
        Returns:
            dict: Tracking results
        """
        try:
            # Set default paths
            if keypoints_json_path is None:
                keypoints_json_path = os.path.join(self.paths['sample_data'], 'flow_image_pair', 'ref_img_keypoints.json')
            if output_dir is None:
                output_dir = self.paths['output']
                
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            if verbose:
                print("🎯 Starting keypoint tracking...")
                print(f"   Reference keypoints: {keypoints_json_path}")
                print(f"   Output directory: {output_dir}")
            
            # Load keypoints
            if verbose:
                print("📍 Loading keypoints...")
            keypoints, original_size = load_keypoints(keypoints_json_path)
            if verbose:
                print(f"   Loaded {len(keypoints)} keypoints")
                print(f"   Original image size: {original_size}")
            
            # Load images
            if verbose:
                print("🖼️  Loading images...")
            ref_img, comp_img = self.load_images(ref_img_path, comp_img_path, target_width)
            if verbose:
                print(f"   Reference image: {ref_img.shape}")
                print(f"   Comparison image: {comp_img.shape}")
                if target_width is not None:
                    print(f"   Resized to target width: {target_width}")
                else:
                    print("   Using original resolution")
            
            # Resize keypoints to match image size
            if verbose:
                print("🔄 Resizing keypoints...")
            resized_keypoints = resize_keypoints(keypoints, original_size, ref_img.shape[:2][::-1])
            
            # Compute optical flow
            if verbose:
                print("🚀 Computing optical flow...")
            start_time = time.time()
            flow = self.client.compute_flow(ref_img, comp_img)
            flow_time = time.time() - start_time
            if verbose:
                print(f"✅ Flow computed in {flow_time:.2f}s")
                print(f"   Flow shape: {flow.shape}")
            
            # Track keypoints
            if verbose:
                print("🎯 Tracking keypoints...")
            tracked_keypoints = self.track_keypoints_with_flow(resized_keypoints, flow)
            
            if verbose:
                print("📊 Keypoint tracking results:")
                for kp in tracked_keypoints:
                    movement = np.sqrt(kp['flow_x']**2 + kp['flow_y']**2)
                    print(f"   - {kp['name']}: moved {movement:.1f} pixels")
            
            # Create visualization
            if verbose:
                print("🎨 Creating visualization...")
            tracking_vis = visualize_tracking_results(ref_img, comp_img, resized_keypoints, tracked_keypoints)
            
            # Save results
            vis_output_path = os.path.join(output_dir, "keypoint_tracking.png")
            Image.fromarray(tracking_vis).save(vis_output_path)
            
            # Save tracking data
            tracked_output_path = os.path.join(output_dir, "tracked_keypoints.json")
            output_data = {
                "original_keypoints": self._convert_numpy_types(resized_keypoints),
                "tracked_keypoints": self._convert_numpy_types(tracked_keypoints),
                "flow_computation_time": float(flow_time),
                "image_dimensions": {
                    "reference": list(ref_img.shape),
                    "comparison": list(comp_img.shape)
                }
            }
            
            with open(tracked_output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            if verbose:
                print(f"✅ Visualization saved: {vis_output_path}")
                print(f"✅ Data saved: {tracked_output_path}")
                print("🎉 Keypoint tracking completed successfully!")
            
            return {
                'success': True,
                'tracked_keypoints': tracked_keypoints,
                'flow_time': flow_time,
                'output_paths': {
                    'visualization': vis_output_path,
                    'data': tracked_output_path
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Error during keypoint tracking: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def reverse_validation(self, tracked_keypoints, original_keypoints, ref_img, comp_img, 
                          output_dir=None, verbose=True):
        """Perform reverse validation by tracking from comp_img back to ref_img.
        
        Args:
            tracked_keypoints: Keypoints detected in comp_img
            original_keypoints: Original keypoints from ref_img
            ref_img: Reference image (numpy array)
            comp_img: Comparison image (numpy array)
            output_dir: Output directory for results
            verbose: Whether to print progress messages
            
        Returns:
            dict: Reverse validation results including comparison metrics
        """
        try:
            if output_dir is None:
                output_dir = self.paths['output']
            
            os.makedirs(output_dir, exist_ok=True)
            
            if verbose:
                print("🔄 Starting reverse validation...")
            
            # Compute reverse optical flow (from comp_img back to ref_img)
            start_time = time.time()
            reverse_flow = self.client.compute_flow(comp_img, ref_img)
            reverse_flow_time = time.time() - start_time
            
            if verbose:
                print(f"✅ Reverse flow computed in {reverse_flow_time:.2f}s")
            
            # Create keypoints from tracked positions for reverse tracking
            reverse_input_keypoints = []
            for i, kp in enumerate(tracked_keypoints):
                reverse_kp = {
                    'id': kp.get('id', i + 1),
                    'name': kp.get('name', f'keypoint_{i+1}'),
                    'x': kp['new_x'],  # Use the tracked position as input
                    'y': kp['new_y'],
                    'coordinates_type': 'image_pixels'
                }
                reverse_input_keypoints.append(reverse_kp)
            
            if verbose:
                print(f"🎯 Reverse tracking {len(reverse_input_keypoints)} keypoints...")
            
            # Track keypoints using reverse flow
            reverse_tracked_keypoints = self.track_keypoints_with_flow(reverse_input_keypoints, reverse_flow)
            
            # Compare with original keypoints
            comparison_results = compare_keypoints(original_keypoints, reverse_tracked_keypoints)
            
            if verbose:
                for result in comparison_results['individual_results']:
                    status = "✅" if result['error_within_threshold'] else "❌"
                    print(f"   {status} {result['keypoint_name']}: error = {result['error_distance_pixels']:.2f} pixels")
            
            # Create validation visualization
            if verbose:
                print("🎨 Creating reverse validation visualization...")
            validation_vis = visualize_reverse_validation_results(
                ref_img, comp_img, original_keypoints, tracked_keypoints, 
                reverse_tracked_keypoints, comparison_results
            )
            
            # Save visualization
            vis_output_path = os.path.join(output_dir, "reverse_validation_visualization.png")
            Image.fromarray(validation_vis).save(vis_output_path)
            
            # Prepare output data
            output_data = {
                "validation_summary": {
                    "total_keypoints": len(original_keypoints),
                    "average_error_pixels": comparison_results['average_error'],
                    "max_error_pixels": comparison_results['max_error'],
                    "min_error_pixels": comparison_results['min_error'],
                    "validation_passed": comparison_results['validation_passed'],
                    "error_threshold_pixels": comparison_results['error_threshold']
                },
                "keypoint_comparisons": comparison_results['individual_results'],
                "reverse_flow_computation_time": float(reverse_flow_time),
                "original_keypoints": self._convert_numpy_types(original_keypoints),
                "forward_tracked_keypoints": self._convert_numpy_types(tracked_keypoints),
                "reverse_tracked_keypoints": self._convert_numpy_types(reverse_tracked_keypoints)
            }
            
            # Save validation results
            validation_output_path = os.path.join(output_dir, "keypoint_reverse_validation.json")
            with open(validation_output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            if verbose:
                print(f"✅ Validation visualization saved: {vis_output_path}")
                print(f"✅ Validation data saved: {validation_output_path}")
                print("🎉 Reverse validation completed successfully!")
                print(f"📊 Average error: {comparison_results['average_error']:.2f} pixels")
                print(f"📊 Max error: {comparison_results['max_error']:.2f} pixels")
                if comparison_results['validation_passed']:
                    print("✅ Validation PASSED - errors within acceptable range")
                else:
                    print("❌ Validation FAILED - some errors exceed threshold")
            
            return {
                'success': True,
                'validation_results': comparison_results,
                'reverse_flow_time': reverse_flow_time,
                'output_paths': {
                    'visualization': vis_output_path,
                    'validation_data': validation_output_path
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Error during reverse validation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


def main():
    """Main function for command line usage."""
    tracker = KeypointTracker()
    result = tracker.run_tracking()
    
    if result['success']:
        print("\n💡 Tracking completed successfully!")
        print("   - Keypoints tracked using optical flow")
        print("   - Results saved to test_data/output/")
    else:
        print(f"\n❌ Tracking failed: {result['error']}")
        return False
    
    return True


if __name__ == "__main__":
    main()
