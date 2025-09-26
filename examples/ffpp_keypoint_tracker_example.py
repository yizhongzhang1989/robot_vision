#!/usr/bin/env python3
"""
FFPPKeypointTracker Example and Test Suite

This example demonstrates the usage of FFPPKeypointTracker with the simplified interface
and bidirectional flow validation features. It serves as both a usage example and
automated test suite.

Code Structure:
- list_cuda_devices(): Lists available CUDA devices for model initialization
- load_sample_data(): Returns (target_img, ref_img, ref_keypoints) tuple
- test_basic_tracking(): Demonstrates core two-step tracking process
- test_bidirectional_validation(): Shows accuracy assessment features
- test_multiple_references(): Multiple reference management demo
- test_resolution_scaling(): Tests tracking across different image resolutions
- test_flow_visualization(): Extracts and visualizes raw optical flow data
- run_performance_benchmark(): Performance comparison between modes

Features demonstrated:
- CUDA device detectioif __name__ == "__main__":
    main()n display
- Basic keypoint tracking with stored references
- Multiple reference image management  
- Bidirectional flow validation for accuracy assessment
- Output visualization and JSON export
- Performance benchmarking

Output Location:
- All results saved to: output/ffpp_keypoint_tracker_example_output/
- Includes visualizations, JSON data, and performance benchmarks

Usage:
    python examples/ffpp_keypoint_tracker_example.py              # Run full test suite
    python examples/ffpp_keypoint_tracker_example.py --devices    # Check CUDA devices only
    python examples/ffpp_keypoint_tracker_example.py -d          # Check CUDA devices (short)
"""

import os
import cv2
import json
import numpy as np
import time
import torch

# Add the parent directory to the path to import core modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.ffpp_keypoint_tracker import FFPPKeypointTracker


def smooth_color_transition(value):
    """
    Create smooth color transition for values from 0 to 1.
    
    Color progression:
    0.0 -> (255,0,0)   Red
    0.25 -> (255,255,0) Yellow  
    0.5 -> (0,255,0)   Green
    0.75 -> (0,255,255) Cyan
    1.0 -> (0,0,255)   Blue
    
    Args:
        value (float): Value between 0 and 1
        
    Returns:
        tuple: (B, G, R) color values for OpenCV
    """
    # Clamp value to [0, 1]
    value = max(0.0, min(1.0, value))
    
    # Define the 5 key colors in RGB format
    colors_rgb = [
        (255, 0, 0),    # Red (0.0)
        (255, 255, 0),  # Yellow (0.25)
        (0, 255, 0),    # Green (0.5)
        (0, 255, 255),  # Cyan (0.75)
        (0, 0, 255)     # Blue (1.0)
    ]
    
    # Scale value to segment index
    scaled_value = value * 4  # 4 segments between 5 colors
    segment_index = int(scaled_value)
    local_t = scaled_value - segment_index
    
    # Handle edge case
    if segment_index >= 4:
        segment_index = 3
        local_t = 1.0
    
    # Get the two colors to interpolate between
    color1_rgb = colors_rgb[segment_index]
    color2_rgb = colors_rgb[segment_index + 1]
    
    # Linear interpolation between the two colors
    r = int(color1_rgb[0] * (1 - local_t) + color2_rgb[0] * local_t)
    g = int(color1_rgb[1] * (1 - local_t) + color2_rgb[1] * local_t)
    b = int(color1_rgb[2] * (1 - local_t) + color2_rgb[2] * local_t)
    
    # Return in BGR format for OpenCV
    return (b, g, r)


def consistency_to_color(consistency_distance, max_distance=5.0):
    """
    Map consistency distance to color using smooth transition.
    
    Args:
        consistency_distance (float): Consistency distance in pixels
        max_distance (float): Maximum distance to map to blue (default: 5.0)
        
    Returns:
        tuple: (B, G, R) color values for OpenCV
    """
    # Normalize consistency distance to [0, 1] range
    normalized_value = consistency_distance / max_distance
    return smooth_color_transition(normalized_value)


def list_cuda_devices():
    """
    List all available CUDA devices that can be used for model initialization.
    
    Returns:
        dict: Dictionary containing CUDA availability info and device details
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': [],
        'current_device': None
    }
    
    if torch.cuda.is_available():
        device_info['device_count'] = torch.cuda.device_count()
        device_info['current_device'] = torch.cuda.current_device()
        
        print(f"🔧 CUDA Device Information:")
        print(f"   CUDA Available: ✅ Yes")
        print(f"   Number of devices: {device_info['device_count']}")
        print(f"   Current device: {device_info['current_device']}")
        print()
        
        for i in range(device_info['device_count']):
            props = torch.cuda.get_device_properties(i)
            device_details = {
                'id': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count
            }
            device_info['devices'].append(device_details)
            
            print(f"   Device {i}: {props.name}")
            print(f"     Total Memory: {device_details['total_memory_gb']:.1f} GB")
            print(f"     Compute Capability: {device_details['compute_capability']}")
            print(f"     Multiprocessors: {device_details['multiprocessor_count']}")
            
            # Show memory usage if device is current
            if i == device_info['current_device']:
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"     Memory Usage: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
            print()
    else:
        print(f"🔧 CUDA Device Information:")
        print(f"   CUDA Available: ❌ No")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   Running on CPU only")
        print()
    
    return device_info


def load_sample_data():
    """
    Load sample data for testing: reference image, target image, and keypoints.
    
    Returns:
        tuple: (target_img, ref_img, ref_keypoints) or None if loading fails
    """
    # Define paths
    ref_image_path = 'sample_data/flow_image_pair/ref_img.jpg'
    comp_image_path = 'sample_data/flow_image_pair/comp_img.jpg'
    ref_keypoints_path = 'sample_data/flow_image_pair/ref_img_keypoints.json'
    
    try:
        # Load images
        ref_img = cv2.imread(ref_image_path)
        target_img = cv2.imread(comp_image_path)
        
        if ref_img is None or target_img is None:
            print("❌ Failed to load sample images - check if files exist")
            return None
        
        # Load keypoints
        with open(ref_keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        
        ref_keypoints = keypoints_data['keypoints']
        
        return target_img, ref_img, ref_keypoints
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return None


def test_basic_tracking():
    """
    Test 1: Basic keypoint tracking functionality
    
    This test demonstrates the core two-step process:
    1. set_reference_image() - Store reference with keypoints
    2. track_keypoints() - Track keypoints in target image
    """
    print("🧪 Test 1: Basic Keypoint Tracking")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    target_img, ref_img, ref_keypoints = load_sample_data()
    
    # ========================================
    # TRACKER INITIALIZATION
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
    
    # ========================================
    # KEYPOINT TRACKING
    # ========================================
    print("\n🎯 Step 1: Setting reference image...")
    ref_result = tracker.set_reference_image(ref_img, ref_keypoints)
    
    if not ref_result['success']:
        print(f"❌ Failed to set reference image: {ref_result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Reference image set with {ref_result['keypoints_count']} keypoints")
    
    print("\n🎯 Step 2: Tracking keypoints in target image...")
    start_time = time.time()
    result = tracker.track_keypoints(target_img)
    elapsed_time = time.time() - start_time
    
    if not result['success']:
        print(f"❌ Keypoint tracking failed: {result.get('error', 'Unknown error')}")
        return False
    
    tracked_count = len(result.get('tracked_keypoints', []))
    print(f"✅ Keypoint tracking successful!")
    print(f"   Time: {elapsed_time:.3f}s")
    print(f"   Tracked: {tracked_count} keypoints")
    print(f"   Processing time: {result.get('total_processing_time', 0):.3f}s")
    
    # ========================================
    # OUTPUT AND VISUALIZATION
    # ========================================
    print("\n📊 Creating outputs...")
    try:
        # Create output directory
        output_dir = 'output/ffpp_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. SAVE JSON RESULTS FIRST
        json_path = os.path.join(output_dir, 'basic_tracking_results.json')
        
        # Write the result directly as returned by track_keypoints()
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Results saved: {json_path}")
        
        # 2. CREATE AND SAVE VISUALIZATION IMAGE
        vis_img = target_img.copy()
        for i, kp in enumerate(result['tracked_keypoints']):
            x, y = int(round(kp['x'])), int(round(kp['y']))
            h, w = vis_img.shape[:2]
            
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(vis_img, str(i+1), (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        vis_path = os.path.join(output_dir, 'basic_tracking_visualization.jpg')
        cv2.imwrite(vis_path, vis_img)
        
        print(f"✅ Visualization saved: {vis_path}")
        
    except Exception as e:
        print(f"⚠️ Output creation failed: {e}")
    
    return True


def test_bidirectional_validation():
    """
    Test 2: Bidirectional flow validation
    
    This test demonstrates the accuracy assessment feature using bidirectional
    flow computation to measure tracking consistency.
    """
    print("\n🧪 Test 2: Bidirectional Flow Validation")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    target_img, ref_img, ref_keypoints = load_sample_data()
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    tracker = FFPPKeypointTracker()
    
    if not tracker.model_loaded:
        print("❌ Failed to load model")
        return False
    
    # ========================================
    # KEYPOINT TRACKING
    # ========================================
    print("\n🎯 Step 1: Setting reference image...")
    tracker.set_reference_image(ref_img, ref_keypoints)
    print(f"✅ Reference image set with {len(ref_keypoints)} keypoints")
    
    print("\n🎯 Step 2: Tracking keypoints with bidirectional validation...")
    start_time = time.time()
    result = tracker.track_keypoints(target_img, bidirectional=True)
    elapsed_time = time.time() - start_time
    
    if not result['success']:
        print(f"❌ Bidirectional tracking failed: {result.get('error', 'Unknown error')}")
        return False
    
    tracked_count = len(result.get('tracked_keypoints', []))
    print(f"✅ Bidirectional tracking successful!")
    print(f"   Time: {elapsed_time:.3f}s")
    print(f"   Tracked: {tracked_count} keypoints")
    print(f"   Processing time: {result.get('total_processing_time', 0):.3f}s")
    
    # Show displacement statistics
    if tracked_count > 0:
        displacements = [(kp.get('displacement_x', 0), kp.get('displacement_y', 0)) 
                        for kp in result['tracked_keypoints']]
        avg_displacement = np.mean([np.sqrt(dx**2 + dy**2) for dx, dy in displacements])
        print(f"   Average displacement: {avg_displacement:.1f} pixels")
    
    # Analyze bidirectional statistics
    if 'bidirectional_stats' in result:
        stats = result['bidirectional_stats']
        print(f"\n📊 Bidirectional Flow Statistics:")
        print(f"   Consistency distance (mean): {stats.get('mean_consistency_distance', 0):.2f} pixels")
        print(f"   Consistency distance (std): {stats.get('std_consistency_distance', 0):.2f} pixels")
        print(f"   Consistency distance (max): {stats.get('max_consistency_distance', 0):.2f} pixels")
        print(f"   High accuracy points (<1.0px): {stats.get('high_accuracy_count', 0)}")
        print(f"   Medium accuracy points (1.0-2.0px): {stats.get('medium_accuracy_count', 0)}")
        print(f"   Low accuracy points (>2.0px): {stats.get('low_accuracy_count', 0)}")
        
        # Calculate accuracy percentage
        total_points = len(result.get('tracked_keypoints', []))
        if total_points > 0:
            high_acc_pct = (stats.get('high_accuracy_count', 0) / total_points) * 100
            print(f"   High accuracy percentage: {high_acc_pct:.1f}%")
    
    # ========================================
    # OUTPUT AND VISUALIZATION
    # ========================================
    print("\n� Creating outputs...")
    try:
        # Create output directory
        output_dir = 'output/ffpp_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. SAVE JSON RESULTS FIRST
        json_path = os.path.join(output_dir, 'bidirectional_validation_results.json')
        
        # Write the result directly as returned by track_keypoints()
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Results saved: {json_path}")
        
        # 2. CREATE AND SAVE VISUALIZATION IMAGE
        vis_img = target_img.copy()
        for i, kp in enumerate(result['tracked_keypoints']):
            x, y = int(round(kp['x'])), int(round(kp['y']))
            h, w = vis_img.shape[:2]
            
            if 0 <= x < w and 0 <= y < h:
                # Use smooth color transition based on consistency distance
                if 'consistency_distance' in kp:
                    consistency = kp['consistency_distance']
                    color = consistency_to_color(consistency, max_distance=5.0)
                else:
                    color = (0, 255, 0)  # Default green for no consistency data
                
                cv2.circle(vis_img, (x, y), 4, color, -1)  # Slightly larger circle
                cv2.putText(vis_img, str(i+1), (x+6, y-6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add color legend to the image
        legend_height = 60
        legend_width = 300
        legend_x = 10
        legend_y = 10
        
        # Create legend background
        cv2.rectangle(vis_img, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(vis_img, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (255, 255, 255), 1)
        
        # Draw color bar
        bar_y = legend_y + 15
        bar_height = 15
        for i in range(legend_width - 20):
            value = i / (legend_width - 20)
            color = smooth_color_transition(value)
            cv2.line(vis_img, (legend_x + 10 + i, bar_y), 
                    (legend_x + 10 + i, bar_y + bar_height), color, 1)
        
        # Add text labels
        cv2.putText(vis_img, "Consistency: 0px", (legend_x + 10, bar_y + bar_height + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis_img, "5px", (legend_x + legend_width - 30, bar_y + bar_height + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        vis_path = os.path.join(output_dir, 'bidirectional_validation_visualization.jpg')
        cv2.imwrite(vis_path, vis_img)
        
        print(f"✅ Visualization saved: {vis_path}")
        
    except Exception as e:
        print(f"⚠️ Output creation failed: {e}")
    
    return True


def test_multiple_references():
    """
    Test 3: Multiple reference image management
    
    This test demonstrates managing multiple reference images with different
    keypoint sets and switching between them.
    """
    print("\n🧪 Test 3: Multiple Reference Management")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    target_img, ref_img, ref_keypoints = load_sample_data()
    
    # Use the loaded keypoints directly for different reference scenarios
    half_keypoints = ref_keypoints[:len(ref_keypoints)//2]  # First half for testing
    
    print(f"✅ Using keypoint sets from loaded data:")
    print(f"   Full set: {len(ref_keypoints)} keypoints")
    print(f"   Half set: {len(half_keypoints)} keypoints")
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    tracker = FFPPKeypointTracker()
    
    if not tracker.model_loaded:
        print("❌ Failed to load model")
        return False
    
    # Set up multiple references
    print("\n📋 Setting up multiple references...")
    
    # Reference 1: Full keypoint set (becomes default)
    tracker.set_reference_image(ref_img, ref_keypoints)
    print(f"   Set default reference: {len(ref_keypoints)} keypoints")
    
    # Reference 2: Half keypoint set
    tracker.set_reference_image(ref_img, half_keypoints, image_name="half_set")
    print(f"   Set 'half_set' reference: {len(half_keypoints)} keypoints")
    
    # Reference 3: Image only (no keypoints)
    tracker.set_reference_image(ref_img, image_name="image_only")
    print(f"   Set 'image_only' reference: 0 keypoints")
    
    print(f"\n📍 Active references: {list(tracker.reference_data.keys())}")
    print(f"   Default reference: '{tracker.default_reference_key}'")
    
    # Test tracking with different references
    print("\n🎯 Testing tracking with different references...")
    
    results = {}
    
    # Track with default reference
    start_time = time.time()
    result_default = tracker.track_keypoints(target_img)
    elapsed_time = time.time() - start_time
    results['default'] = result_default
    print(f"   Default: {elapsed_time:.3f}s - {len(result_default.get('tracked_keypoints', []))} points")
    
    # Track with specific references
    for ref_name in ["half_set", "image_only"]:
        start_time = time.time()
        result = tracker.track_keypoints(target_img, reference_name=ref_name)
        elapsed_time = time.time() - start_time
        results[ref_name] = result
        print(f"   {ref_name}: {elapsed_time:.3f}s - {len(result.get('tracked_keypoints', []))} points")
    
    # Test reference management
    print("\n🗑️ Testing reference removal...")
    print(f"   Before removal: {list(tracker.reference_data.keys())}")
    
    tracker.remove_reference_image("image_only")
    print(f"   After removing 'image_only': {list(tracker.reference_data.keys())}")
    
    tracker.remove_reference_image(None)  # Remove default
    print(f"   After removing default: {list(tracker.reference_data.keys())}")
    
    # Save multiple reference results
    print("\n💾 Saving multiple reference results...")
    try:
        output_dir = 'output/ffpp_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, 'multiple_references_results.json')
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': 'multiple_references',
            'reference_tests': {
                name: {
                    'success': result.get('success', False),
                    'keypoints_count': len(result.get('tracked_keypoints', [])),
                    'tracked_keypoints': result.get('tracked_keypoints', [])
                }
                for name, result in results.items()
            },
            'final_references': list(tracker.reference_data.keys())
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Multiple reference results saved: {json_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
    
    print(f"\n✅ Multiple reference test completed")
    print(f"   Final references: {list(tracker.reference_data.keys())}")
    print(f"   Current default: '{tracker.default_reference_key}'")
    
    return True


def test_resolution_scaling():
    """
    Test 4: Resolution scaling compatibility
    
    This test demonstrates tracking keypoints across different image resolutions.
    The reference image and keypoints are scaled to 2/3 of the original size,
    then tracked on the full-size target image to test scale robustness.
    """
    print("\n🧪 Test 4: Resolution Scaling")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    target_img, ref_img, ref_keypoints = load_sample_data()
    
    # Get original dimensions
    orig_h, orig_w = ref_img.shape[:2]
    print(f"✅ Original image size: {orig_w}x{orig_h}")
    
    # Calculate 2/3 scale dimensions
    scale_factor = 2.0 / 3.0
    scaled_w = int(orig_w * scale_factor)
    scaled_h = int(orig_h * scale_factor)
    
    print(f"✅ Scaled reference size: {scaled_w}x{scaled_h} (scale factor: {scale_factor:.3f})")
    
    # Resize reference image
    scaled_ref_img = cv2.resize(ref_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    
    # Scale reference keypoints
    scaled_ref_keypoints = []
    for kp in ref_keypoints:
        scaled_kp = {
            'x': kp['x'] * scale_factor,
            'y': kp['y'] * scale_factor,
            'id': kp.get('id', len(scaled_ref_keypoints))
        }
        scaled_ref_keypoints.append(scaled_kp)
    
    print(f"✅ Scaled {len(scaled_ref_keypoints)} keypoints to new resolution")
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    tracker = FFPPKeypointTracker()
    
    if not tracker.model_loaded:
        print("❌ Failed to load model")
        return False
    
    # ========================================
    # CROSS-SCALE TRACKING
    # ========================================
    print("\n🎯 Step 1: Setting scaled reference image...")
    ref_result = tracker.set_reference_image(scaled_ref_img, scaled_ref_keypoints)
    
    if not ref_result['success']:
        print(f"❌ Failed to set reference image: {ref_result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Scaled reference image set with {ref_result['keypoints_count']} keypoints")
    
    print("\n🎯 Step 2: Tracking keypoints in full-size target image...")
    start_time = time.time()
    result = tracker.track_keypoints(target_img)
    elapsed_time = time.time() - start_time
    
    if not result['success']:
        print(f"❌ Cross-scale tracking failed: {result.get('error', 'Unknown error')}")
        return False
    
    tracked_count = len(result.get('tracked_keypoints', []))
    print(f"✅ Cross-scale tracking successful!")
    print(f"   Time: {elapsed_time:.3f}s")
    print(f"   Tracked: {tracked_count} keypoints")
    print(f"   Processing time: {result.get('total_processing_time', 0):.3f}s")
    
    # ========================================
    # OUTPUT AND VISUALIZATION
    # ========================================
    print("\n📊 Creating outputs...")
    try:
        # Create output directory
        output_dir = 'output/ffpp_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. SAVE JSON RESULTS FIRST
        json_path = os.path.join(output_dir, 'resolution_scaling_results.json')
        
        # Enhance result with scale information
        enhanced_result = result.copy()
        enhanced_result['scale_test_info'] = {
            'original_size': {'width': orig_w, 'height': orig_h},
            'scaled_reference_size': {'width': scaled_w, 'height': scaled_h},
            'scale_factor': scale_factor,
            'target_size': {'width': target_img.shape[1], 'height': target_img.shape[0]}
        }
        
        with open(json_path, 'w') as f:
            json.dump(enhanced_result, f, indent=2)
        
        print(f"✅ Results saved: {json_path}")
        
        # 2. CREATE COMPARISON VISUALIZATION
        # Create side-by-side visualization showing scaled reference and full target
        vis_height = max(scaled_h, target_img.shape[0])
        vis_width = scaled_w + target_img.shape[1] + 20  # 20px gap
        vis_img = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Place scaled reference image on the left
        vis_img[:scaled_h, :scaled_w] = scaled_ref_img
        
        # Place target image on the right
        target_start_x = scaled_w + 20
        vis_img[:target_img.shape[0], target_start_x:target_start_x + target_img.shape[1]] = target_img
        
        # Draw scaled reference keypoints on the left
        for i, kp in enumerate(scaled_ref_keypoints):
            x, y = int(round(kp['x'])), int(round(kp['y']))
            if 0 <= x < scaled_w and 0 <= y < scaled_h:
                cv2.circle(vis_img, (x, y), 3, (255, 0, 0), -1)  # Blue for reference
                cv2.putText(vis_img, str(i+1), (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw tracked keypoints on the right
        for i, kp in enumerate(result['tracked_keypoints']):
            x, y = int(round(kp['x'])) + target_start_x, int(round(kp['y']))
            target_h, target_w = target_img.shape[:2]
            if target_start_x <= x < target_start_x + target_w and 0 <= y < target_h:
                cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)  # Green for tracked
                cv2.putText(vis_img, str(i+1), (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add labels
        cv2.putText(vis_img, f"Reference ({scaled_w}x{scaled_h})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, f"Target ({target_img.shape[1]}x{target_img.shape[0]})", 
                   (target_start_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add scale factor info
        cv2.putText(vis_img, f"Scale factor: {scale_factor:.3f}", (10, vis_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        vis_path = os.path.join(output_dir, 'resolution_scaling_visualization.jpg')
        cv2.imwrite(vis_path, vis_img)
        
        print(f"✅ Visualization saved: {vis_path}")
        
        # 3. CREATE SINGLE TARGET IMAGE WITH TRACKING RESULTS
        target_vis = target_img.copy()
        for i, kp in enumerate(result['tracked_keypoints']):
            x, y = int(round(kp['x'])), int(round(kp['y']))
            h, w = target_vis.shape[:2]
            
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(target_vis, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(target_vis, str(i+1), (x+6, y-6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add scale info overlay
        overlay_text = f"Tracked from {scale_factor:.1%} scale reference"
        cv2.putText(target_vis, overlay_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        single_vis_path = os.path.join(output_dir, 'resolution_scaling_target_result.jpg')
        cv2.imwrite(single_vis_path, target_vis)
        
        print(f"✅ Target result saved: {single_vis_path}")
        
    except Exception as e:
        print(f"⚠️ Output creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return True


def test_flow_visualization():
    """
    Test 5: Flow visualization with bidirectional flow
    
    This test demonstrates extracting and visualizing raw optical flow data
    from the tracker. Tests both forward and reverse flow visualization
    using bidirectional flow computation.
    """
    print("\n🧪 Test 5: Flow Visualization")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    target_img, ref_img, ref_keypoints = load_sample_data()
    
    print(f"✅ Loaded images: ref {ref_img.shape}, target {target_img.shape}")
    print(f"✅ Reference keypoints: {len(ref_keypoints)}")
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    tracker = FFPPKeypointTracker()
    
    if not tracker.model_loaded:
        print("❌ Failed to load model")
        return False
    
    # ========================================
    # FLOW COMPUTATION AND VISUALIZATION
    # ========================================
    print("\n🎯 Step 1: Setting reference image...")
    ref_result = tracker.set_reference_image(ref_img, ref_keypoints)
    
    if not ref_result['success']:
        print(f"❌ Failed to set reference image: {ref_result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Reference image set with {ref_result['keypoints_count']} keypoints")
    
    print("\n🎯 Step 2: Computing bidirectional flow with return_flow=True...")
    start_time = time.time()
    result = tracker.track_keypoints(target_img, bidirectional=True, return_flow=True)
    elapsed_time = time.time() - start_time
    
    if not result['success']:
        print(f"❌ Flow computation failed: {result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Bidirectional flow computation successful!")
    print(f"   Time: {elapsed_time:.3f}s")
    print(f"   Tracked keypoints: {len(result['tracked_keypoints'])}")
    
    # Extract flow data
    flow_data = result['flow_data']
    forward_flow = flow_data['forward_flow']
    reverse_flow = flow_data['reverse_flow']
    
    print(f"   Forward flow shape: {forward_flow.shape}")
    print(f"   Reverse flow shape: {reverse_flow.shape if reverse_flow is not None else 'None'}")
    
    # Flow statistics
    forward_magnitude = np.sqrt(forward_flow[:, :, 0]**2 + forward_flow[:, :, 1]**2)
    print(f"   Forward flow magnitude: mean={forward_magnitude.mean():.2f}, max={forward_magnitude.max():.2f}")
    
    if reverse_flow is not None:
        reverse_magnitude = np.sqrt(reverse_flow[:, :, 0]**2 + reverse_flow[:, :, 1]**2)
        print(f"   Reverse flow magnitude: mean={reverse_magnitude.mean():.2f}, max={reverse_magnitude.max():.2f}")
    
    # ========================================
    # FLOW VISUALIZATION
    # ========================================
    print("\n🎨 Creating flow visualizations...")
    try:
        # Import flow visualization utilities
        import sys
        import os
        sys.path.append('ThirdParty/FlowFormerPlusPlusServer')
        from core.utils import flow_viz
        
        # Create output directory
        output_dir = 'output/ffpp_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. SAVE JSON RESULTS FIRST
        json_path = os.path.join(output_dir, 'flow_visualization_results.json')
        
        # Create a serializable version of the results (without numpy arrays)
        json_result = {
            'success': result['success'],
            'tracked_keypoints': result['tracked_keypoints'],
            'flow_computation_time': result['flow_computation_time'],
            'reverse_flow_computation_time': result['reverse_flow_computation_time'],
            'total_processing_time': result['total_processing_time'],
            'bidirectional_enabled': result['bidirectional_enabled'],
            'bidirectional_stats': result['bidirectional_stats'],
            'forward_flow_shape': list(forward_flow.shape),
            'reverse_flow_shape': list(reverse_flow.shape) if reverse_flow is not None else None,
            'flow_statistics': {
                'forward_flow': {
                    'mean_magnitude': float(forward_magnitude.mean()),
                    'max_magnitude': float(forward_magnitude.max()),
                    'std_magnitude': float(forward_magnitude.std())
                },
                'reverse_flow': {
                    'mean_magnitude': float(reverse_magnitude.mean()),
                    'max_magnitude': float(reverse_magnitude.max()),
                    'std_magnitude': float(reverse_magnitude.std())
                } if reverse_flow is not None else None
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"✅ Results saved: {json_path}")
        
        # 2. VISUALIZE FORWARD FLOW
        print("   Creating forward flow visualization...")
        forward_flow_img = flow_viz.flow_to_image(forward_flow)
        forward_vis_path = os.path.join(output_dir, 'forward_flow_visualization.jpg')
        cv2.imwrite(forward_vis_path, forward_flow_img[:, :, [2, 1, 0]])  # RGB to BGR for OpenCV
        print(f"✅ Forward flow saved: {forward_vis_path}")
        
        # 3. VISUALIZE REVERSE FLOW
        if reverse_flow is not None:
            print("   Creating reverse flow visualization...")
            reverse_flow_img = flow_viz.flow_to_image(reverse_flow)
            reverse_vis_path = os.path.join(output_dir, 'reverse_flow_visualization.jpg')
            cv2.imwrite(reverse_vis_path, reverse_flow_img[:, :, [2, 1, 0]])  # RGB to BGR for OpenCV
            print(f"✅ Reverse flow saved: {reverse_vis_path}")
        
        # 4. CREATE COMBINED VISUALIZATION
        print("   Creating combined flow visualization...")
        
        # Create side-by-side comparison
        h, w = forward_flow_img.shape[:2]
        if reverse_flow is not None:
            combined_width = w * 2 + 20  # 20px gap
            combined_img = np.zeros((h, combined_width, 3), dtype=np.uint8)
            
            # Place forward flow on the left
            combined_img[:h, :w] = forward_flow_img
            
            # Place reverse flow on the right
            combined_img[:h, w+20:w*2+20] = reverse_flow_img
            
            # Add labels
            cv2.putText(combined_img, "Forward Flow (Ref->Target)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_img, "Reverse Flow (Target->Ref)", (w+30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add flow statistics
            cv2.putText(combined_img, f"Mean: {forward_magnitude.mean():.1f}px", (10, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(combined_img, f"Max: {forward_magnitude.max():.1f}px", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(combined_img, f"Mean: {reverse_magnitude.mean():.1f}px", (w+30, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(combined_img, f"Max: {reverse_magnitude.max():.1f}px", (w+30, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            combined_img = forward_flow_img
            cv2.putText(combined_img, "Forward Flow Only", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        combined_vis_path = os.path.join(output_dir, 'combined_flow_visualization.jpg')
        cv2.imwrite(combined_vis_path, combined_img[:, :, [2, 1, 0]])  # RGB to BGR for OpenCV
        print(f"✅ Combined flow saved: {combined_vis_path}")
        
        # 5. CREATE KEYPOINTS OVERLAY ON FLOW
        print("   Creating keypoints overlay on flow...")
        
        # Overlay keypoints on forward flow
        keypoints_flow_img = forward_flow_img.copy()
        for i, kp in enumerate(result['tracked_keypoints']):
            # Show original reference keypoint position
            if i < len(ref_keypoints):
                orig_x, orig_y = int(ref_keypoints[i]['x']), int(ref_keypoints[i]['y'])
                if 0 <= orig_x < w and 0 <= orig_y < h:
                    cv2.circle(keypoints_flow_img, (orig_x, orig_y), 3, (255, 255, 255), -1)
                    cv2.circle(keypoints_flow_img, (orig_x, orig_y), 4, (0, 0, 0), 1)
            
            # Show tracked keypoint position
            track_x, track_y = int(kp['x']), int(kp['y'])
            if 0 <= track_x < w and 0 <= track_y < h:
                cv2.circle(keypoints_flow_img, (track_x, track_y), 3, (0, 255, 0), -1)
                cv2.circle(keypoints_flow_img, (track_x, track_y), 4, (0, 0, 0), 1)
                cv2.putText(keypoints_flow_img, str(i+1), (track_x+5, track_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add legend
        cv2.putText(keypoints_flow_img, "White: Reference keypoints", (10, h-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(keypoints_flow_img, "Green: Tracked keypoints", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        keypoints_overlay_path = os.path.join(output_dir, 'flow_keypoints_overlay.jpg')
        cv2.imwrite(keypoints_overlay_path, keypoints_flow_img[:, :, [2, 1, 0]])  # RGB to BGR for OpenCV
        print(f"✅ Keypoints overlay saved: {keypoints_overlay_path}")
        
    except Exception as e:
        print(f"⚠️ Flow visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    return True


def run_performance_benchmark():
    """
    Performance benchmark: Measure and compare different tracking modes
    """
    print("\n🧪 Performance Benchmark")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    target_img, ref_img, ref_keypoints = load_sample_data()
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    tracker = FFPPKeypointTracker()
    
    if not tracker.model_loaded:
        print("❌ Failed to load model")
        return False
    
    # Set reference once
    tracker.set_reference_image(ref_img, ref_keypoints)
    
    # Benchmark different modes
    print("\n⏱️ Running performance benchmarks...")
    
    benchmarks = {}
    num_runs = 5
    
    # Benchmark 1: Standard tracking
    print(f"   Testing standard tracking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start_time = time.time()
        result = tracker.track_keypoints(target_img)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        if not result['success']:
            print(f"❌ Run {i+1} failed")
            return False
    
    benchmarks['standard'] = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'keypoints_count': len(result.get('tracked_keypoints', []))
    }
    
    # Benchmark 2: Bidirectional tracking
    print(f"   Testing bidirectional tracking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start_time = time.time()
        result = tracker.track_keypoints(target_img, bidirectional=True)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        if not result['success']:
            print(f"❌ Bidirectional run {i+1} failed")
            return False
    
    benchmarks['bidirectional'] = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'keypoints_count': len(result.get('tracked_keypoints', [])),
        'consistency_stats': result.get('bidirectional_stats', {})
    }
    
    # Display results
    print(f"\n📊 Performance Benchmark Results:")
    print(f"   Standard tracking:")
    print(f"     Mean time: {benchmarks['standard']['mean_time']:.3f}s ± {benchmarks['standard']['std_time']:.3f}s")
    print(f"     Range: {benchmarks['standard']['min_time']:.3f}s - {benchmarks['standard']['max_time']:.3f}s")
    print(f"     Keypoints: {benchmarks['standard']['keypoints_count']}")
    
    print(f"   Bidirectional tracking:")
    print(f"     Mean time: {benchmarks['bidirectional']['mean_time']:.3f}s ± {benchmarks['bidirectional']['std_time']:.3f}s")
    print(f"     Range: {benchmarks['bidirectional']['min_time']:.3f}s - {benchmarks['bidirectional']['max_time']:.3f}s")
    print(f"     Keypoints: {benchmarks['bidirectional']['keypoints_count']}")
    
    # Calculate overhead
    overhead = benchmarks['bidirectional']['mean_time'] - benchmarks['standard']['mean_time']
    overhead_pct = (overhead / benchmarks['standard']['mean_time']) * 100
    print(f"   Bidirectional overhead: +{overhead:.3f}s ({overhead_pct:.1f}%)")
    
    # Save benchmark results
    try:
        output_dir = 'output/ffpp_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, 'performance_benchmark.json')
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': 'performance_benchmark',
            'num_runs': num_runs,
            'benchmarks': benchmarks,
            'overhead_analysis': {
                'bidirectional_overhead_seconds': overhead,
                'bidirectional_overhead_percent': overhead_pct
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Benchmark results saved: {json_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to save benchmark: {e}")
    
    return True


def main():
    """
    Main function: Run complete test suite
    """
    print("🔥 FFPPKeypointTracker Example & Test Suite 🔥")
    print("=" * 60)
    print("This example demonstrates the simplified FFPPKeypointTracker interface")
    print("with bidirectional flow validation and multiple reference management.")
    print("=" * 60)
    
    # List available CUDA devices
    device_info = list_cuda_devices()
    
    # Check if sample data exists
    if not os.path.exists('sample_data/flow_image_pair/ref_img.jpg'):
        print("❌ Sample data not found!")
        print("   Please ensure sample_data/flow_image_pair/ contains:")
        print("   - ref_img.jpg")
        print("   - comp_img.jpg") 
        print("   - ref_img_keypoints.json")
        return
    
    # Run test suite
    tests = [
        ("Basic Tracking", test_basic_tracking),
        ("Bidirectional Validation", test_bidirectional_validation),
        ("Multiple References", test_multiple_references),
        ("Resolution Scaling", test_resolution_scaling),
        ("Flow Visualization", test_flow_visualization),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n" + "="*60)
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"🎯 Test Suite Summary")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Total time: {total_time:.2f}s")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed successfully!")
        print("   Check the 'output/ffpp_keypoint_tracker_example_output/' directory for generated results and visualizations.")
    else:
        print("⚠️  Some tests failed - check output above for details")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
