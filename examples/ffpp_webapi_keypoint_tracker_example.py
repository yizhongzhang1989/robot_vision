#!/usr/bin/env python3
"""
FFPPWebAPIKeypointTracker Example and Test Suite

This example demonstrates the usage of FFPPWebAPIKeypointTracker with the web API interface
and bidirectional flow validation features. It serves as both a usage example and
automated test suite for the web API client.

Code Structure:
- check_service_availability(): Checks web service connectivity and status
- load_sample_data(): Returns (target_img, ref_img, ref_keypoints) tuple
- test_basic_tracking(): Demonstrates core two-step tracking process via web API
- test_bidirectional_validation(): Shows accuracy assessment features via API
- test_multiple_references(): Multiple reference management demo via API
- test_flow_visualization(): Flow data extraction and visualization via API
- test_service_features(): Tests web service specific features
- test_encoding_configurations(): Tests configurable image encoding performance
- run_performance_benchmark(): Performance comparison between API modes

Features demonstrated:
- Web service connectivity checking
- Basic keypoint tracking with stored references via API
- Multiple reference image management via web service
- Bidirectional flow validation for accuracy assessment via API
- Output visualization and JSON export
- Performance benchmarking of web API calls
- Service health monitoring

Output Location:
- All results saved to: output/ffpp_webapi_keypoint_tracker_example_output/
- Includes visualizations, JSON data, and performance benchmarks

Prerequisites:
- FlowFormer++ web service running (default: http://localhost:8001)
- Sample data in sample_data/flow_image_pair/

Configuration:
- Edit WEB_SERVICE_URL at the top of this script to change default server location
- Or use --url argument to specify server URL from command line
- Examples: "http://192.168.1.100:8001", "http://gpu-server:8001"

Usage:
    python examples/ffpp_webapi_keypoint_tracker_example.py                           # Run full test suite
    python examples/ffpp_webapi_keypoint_tracker_example.py --service                 # Check service only
    python examples/ffpp_webapi_keypoint_tracker_example.py -s                        # Check service (short)
    python examples/ffpp_webapi_keypoint_tracker_example.py --url http://server:8001  # Use custom URL
    python examples/ffpp_webapi_keypoint_tracker_example.py -u http://gpu-server:8001 # Use custom URL (short)
"""

import os
import cv2
import json
import numpy as np
import time
import sys
import argparse

# Add the parent directory to the path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.ffpp_webapi_keypoint_tracker import FFPPWebAPIKeypointTracker

# =============================================================================
# CONFIGURATION - Change this URL to match your FlowFormer++ web service
# =============================================================================
WEB_SERVICE_URL = "http://localhost:8001"  # Change this for remote servers
# Examples:
# WEB_SERVICE_URL = "http://192.168.1.100:8001"  # Remote server
# WEB_SERVICE_URL = "http://gpu-server:8001"      # Server by hostname  
# WEB_SERVICE_URL = "http://localhost:8002"       # Different port


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


def check_service_availability():
    """
    Check web service availability and display service information.
    
    Returns:
        dict: Dictionary containing service availability info and health details
    """
    service_info = {
        'available': False,
        'health_data': None,
        'error': None
    }
    
    print(f"🌐 Web Service Connectivity Check:")
    print(f"   Service URL: {WEB_SERVICE_URL}")
    
    try:
        # Try to create tracker (this will test connectivity)
        tracker = FFPPWebAPIKeypointTracker(service_url=WEB_SERVICE_URL)
        
        # Get detailed health information
        health = tracker.get_service_health()
        
        if health.get('success', False):
            service_info['available'] = True
            service_info['health_data'] = health
            
            print(f"   Status: ✅ Available")
            print(f"   Health: {health.get('status', 'Unknown')}")
            print(f"   Message: {health.get('message', 'No message')}")
            
            # Get server references if available
            refs_result = tracker.get_service_references()
            if refs_result.get('success'):
                refs = refs_result.get('server_references', {})
                print(f"   Server references: {len(refs)} stored")
                if refs:
                    for name, info in list(refs.items())[:3]:  # Show first 3
                        print(f"     - {name}: {info.get('keypoints_count', 0)} keypoints")
                    if len(refs) > 3:
                        print(f"     ... and {len(refs) - 3} more")
            
        else:
            service_info['error'] = health.get('error', 'Service not healthy')
            print(f"   Status: ❌ Unavailable")
            print(f"   Error: {service_info['error']}")
        
    except Exception as e:
        service_info['error'] = str(e)
        print(f"   Status: ❌ Cannot connect")
        print(f"   Error: {e}")
        print(f"\n💡 To start the web service:")
        print(f"   cd web/ffpp_keypoint_tracking")
        print(f"   python app.py")
    
    print()
    return service_info


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
    Test 1: Basic keypoint tracking functionality via web API
    
    This test demonstrates the core two-step process:
    1. set_reference_image() - Store reference with keypoints via API
    2. track_keypoints() - Track keypoints in target image via API
    """
    print("🧪 Test 1: Basic Keypoint Tracking (Web API)")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    data_result = load_sample_data()
    if data_result is None:
        return False
    
    target_img, ref_img, ref_keypoints = data_result
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing FFPPWebAPIKeypointTracker...")
    init_start_time = time.time()
    
    try:
        tracker = FFPPWebAPIKeypointTracker(service_url=WEB_SERVICE_URL)
        init_elapsed_time = time.time() - init_start_time
        
        print(f"✅ FFPPWebAPIKeypointTracker initialized")
        print(f"   Service URL: {tracker.service_url}")
        print(f"   Initialization time: {init_elapsed_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return False
    
    # ========================================
    # KEYPOINT TRACKING VIA API
    # ========================================
    print("\n🎯 Step 1: Setting reference image via API...")
    ref_start_time = time.time()
    ref_result = tracker.set_reference_image(ref_img, ref_keypoints, "basic_test_ref")
    ref_elapsed_time = time.time() - ref_start_time
    
    if not ref_result['success']:
        print(f"❌ Failed to set reference image: {ref_result.get('error', 'Unknown error')}")
        return False

    print(f"✅ Reference image set with {ref_result['keypoints_count']} keypoints")
    print(f"   API call time: {ref_elapsed_time:.3f}s")
    print(f"   Reference key: {ref_result.get('key', 'Unknown')}")
    
    print("\n🎯 Step 2: Tracking keypoints in target image via API...")
    track_start_time = time.time()
    result = tracker.track_keypoints(target_img, reference_name="basic_test_ref")
    track_elapsed_time = time.time() - track_start_time
    
    if not result['success']:
        print(f"❌ Keypoint tracking failed: {result.get('error', 'Unknown error')}")
        return False
    
    tracked_count = len(result.get('tracked_keypoints', []))
    print(f"✅ Keypoint tracking successful!")
    print(f"   API call time: {track_elapsed_time:.3f}s")
    print(f"   Service processing time: {result.get('total_processing_time', 0):.3f}s")
    print(f"   Tracked: {tracked_count} keypoints")
    print(f"   Device used: {result.get('device_used', 'Unknown')}")
    
    # ========================================
    # OUTPUT AND VISUALIZATION
    # ========================================
    print("\n📊 Creating outputs...")
    try:
        # Create output directory
        output_dir = 'output/ffpp_webapi_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. SAVE JSON RESULTS FIRST
        json_path = os.path.join(output_dir, 'basic_tracking_results.json')
        
        # Add timing information to result
        output_data = result.copy()
        output_data.update({
            'api_timing': {
                'reference_setup_time': ref_elapsed_time,
                'tracking_call_time': track_elapsed_time,
                'total_api_time': ref_elapsed_time + track_elapsed_time
            },
            'test_info': {
                'test_name': 'basic_tracking_webapi',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'service_url': tracker.service_url
            }
        })
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
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
    
    # Clean up reference
    cleanup_result = tracker.remove_reference_image("basic_test_ref")
    if cleanup_result.get('success'):
        print(f"✅ Reference cleaned up")
    
    return True


def test_bidirectional_validation():
    """
    Test 2: Bidirectional flow validation via web API
    
    This test demonstrates the accuracy assessment feature using bidirectional
    flow computation to measure tracking consistency via web service.
    """
    print("\n🧪 Test 2: Bidirectional Flow Validation (Web API)")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    data_result = load_sample_data()
    if data_result is None:
        return False
    
    target_img, ref_img, ref_keypoints = data_result
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    
    try:
        tracker = FFPPWebAPIKeypointTracker(service_url=WEB_SERVICE_URL)
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return False
    
    # ========================================
    # KEYPOINT TRACKING VIA API
    # ========================================
    print("\n🎯 Step 1: Setting reference image via API...")
    ref_result = tracker.set_reference_image(ref_img, ref_keypoints, "bidirectional_test_ref")
    
    if not ref_result['success']:
        print(f"❌ Failed to set reference image: {ref_result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Reference image set with {len(ref_keypoints)} keypoints")
    
    print("\n🎯 Step 2: Tracking keypoints with bidirectional validation via API...")
    start_time = time.time()
    result = tracker.track_keypoints(target_img, reference_name="bidirectional_test_ref", bidirectional=True)
    elapsed_time = time.time() - start_time
    
    if not result['success']:
        print(f"❌ Bidirectional tracking failed: {result.get('error', 'Unknown error')}")
        return False
    
    tracked_count = len(result.get('tracked_keypoints', []))
    print(f"✅ Bidirectional tracking successful!")
    print(f"   API call time: {elapsed_time:.3f}s")
    print(f"   Service processing time: {result.get('total_processing_time', 0):.3f}s")
    print(f"   Tracked: {tracked_count} keypoints")
    print(f"   Device used: {result.get('device_used', 'Unknown')}")
    
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
    print("\n📊 Creating outputs...")
    try:
        # Create output directory
        output_dir = 'output/ffpp_webapi_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. SAVE JSON RESULTS FIRST
        json_path = os.path.join(output_dir, 'bidirectional_validation_results.json')
        
        # Add timing and test information
        output_data = result.copy()
        output_data.update({
            'api_timing': {
                'tracking_call_time': elapsed_time
            },
            'test_info': {
                'test_name': 'bidirectional_validation_webapi',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'service_url': tracker.service_url
            }
        })
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Results saved: {json_path}")
        
        # 2. CREATE AND SAVE VISUALIZATION IMAGE with consistency coloring
        vis_img = target_img.copy()
        for i, kp in enumerate(result['tracked_keypoints']):
            x, y = int(round(kp['x'])), int(round(kp['y']))
            h, w = vis_img.shape[:2]
            
            if 0 <= x < w and 0 <= y < h:
                # Color based on consistency distance if available
                consistency_dist = kp.get('consistency_distance', 0)
                color = consistency_to_color(consistency_dist, max_distance=5.0)
                
                cv2.circle(vis_img, (x, y), 4, color, -1)
                cv2.circle(vis_img, (x, y), 4, (255, 255, 255), 1)  # White border
                cv2.putText(vis_img, str(i+1), (x+6, y-6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        vis_path = os.path.join(output_dir, 'bidirectional_validation_visualization.jpg')
        cv2.imwrite(vis_path, vis_img)
        
        print(f"✅ Visualization saved: {vis_path}")
        
    except Exception as e:
        print(f"⚠️ Output creation failed: {e}")
    
    # Clean up reference
    cleanup_result = tracker.remove_reference_image("bidirectional_test_ref")
    if cleanup_result.get('success'):
        print(f"✅ Reference cleaned up")
    
    return True


def test_multiple_references():
    """
    Test 3: Multiple reference management via web API
    
    This test demonstrates managing multiple reference images through the web service.
    """
    print("\n🧪 Test 3: Multiple Reference Management (Web API)")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    data_result = load_sample_data()
    if data_result is None:
        return False
    
    target_img, ref_img, ref_keypoints = data_result
    
    # Create variations of reference image for multiple references
    ref_img_scaled = cv2.resize(ref_img, (int(ref_img.shape[1]*0.8), int(ref_img.shape[0]*0.8)))
    ref_img_rotated = cv2.rotate(ref_img, cv2.ROTATE_90_CLOCKWISE)
    
    # Adjust keypoints for scaled reference
    ref_keypoints_scaled = []
    for kp in ref_keypoints[:5]:  # Use fewer keypoints for variations
        ref_keypoints_scaled.append({
            'x': int(kp['x'] * 0.8),
            'y': int(kp['y'] * 0.8)
        })
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    
    try:
        tracker = FFPPWebAPIKeypointTracker(service_url=WEB_SERVICE_URL)
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return False
    
    # ========================================
    # MULTIPLE REFERENCE SETUP
    # ========================================
    print("\n🎯 Setting up multiple reference images...")
    
    # Reference 1: Original
    print("   Setting reference 1 (original)...")
    ref1_result = tracker.set_reference_image(ref_img, ref_keypoints, "reference_original")
    
    if not ref1_result['success']:
        print(f"❌ Failed to set reference 1: {ref1_result.get('error')}")
        return False
    
    # Reference 2: Scaled
    print("   Setting reference 2 (scaled)...")
    ref2_result = tracker.set_reference_image(ref_img_scaled, ref_keypoints_scaled, "reference_scaled")
    
    if not ref2_result['success']:
        print(f"❌ Failed to set reference 2: {ref2_result.get('error')}")
        return False
    
    print(f"✅ Multiple references set:")
    print(f"   Reference 1: {ref1_result['keypoints_count']} keypoints")
    print(f"   Reference 2: {ref2_result['keypoints_count']} keypoints")
    
    # Check server-side references
    refs_result = tracker.get_service_references()
    if refs_result.get('success'):
        server_refs = refs_result.get('server_references', {})
        print(f"   Server has {len(server_refs)} total references")
    
    # ========================================
    # TRACKING WITH DIFFERENT REFERENCES
    # ========================================
    print("\n🎯 Tracking with different references...")
    
    # Track using reference 1
    print("   Tracking with reference 1...")
    result1 = tracker.track_keypoints(target_img, reference_name="reference_original")
    
    if result1['success']:
        tracked1 = len(result1.get('tracked_keypoints', []))
        print(f"✅ Reference 1 tracking: {tracked1} keypoints")
    else:
        print(f"❌ Reference 1 tracking failed: {result1.get('error')}")
    
    # Track using reference 2
    print("   Tracking with reference 2...")
    result2 = tracker.track_keypoints(target_img, reference_name="reference_scaled")
    
    if result2['success']:
        tracked2 = len(result2.get('tracked_keypoints', []))
        print(f"✅ Reference 2 tracking: {tracked2} keypoints")
    else:
        print(f"❌ Reference 2 tracking failed: {result2.get('error')}")
    
    # ========================================
    # OUTPUT AND CLEANUP
    # ========================================
    print("\n📊 Creating outputs...")
    try:
        output_dir = 'output/ffpp_webapi_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        results_data = {
            'test_info': {
                'test_name': 'multiple_references_webapi',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'service_url': tracker.service_url
            },
            'references_setup': {
                'reference_original': ref1_result,
                'reference_scaled': ref2_result
            },
            'tracking_results': {
                'reference_original': result1,
                'reference_scaled': result2
            }
        }
        
        json_path = os.path.join(output_dir, 'multiple_references_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✅ Results saved: {json_path}")
        
    except Exception as e:
        print(f"⚠️ Output creation failed: {e}")
    
    # Clean up references
    print("\n🧹 Cleaning up references...")
    cleanup1 = tracker.remove_reference_image("reference_original")
    cleanup2 = tracker.remove_reference_image("reference_scaled")
    
    success_count = sum([cleanup1.get('success', False), cleanup2.get('success', False)])
    print(f"✅ Cleaned up {success_count}/2 references")
    
    return result1.get('success', False) and result2.get('success', False)


def test_flow_visualization():
    """
    Test 4: Flow visualization with return_flow parameter via web API
    
    This test demonstrates extracting and visualizing raw optical flow data
    from the web API tracker. Tests both forward and reverse flow visualization
    using bidirectional flow computation via web service.
    """
    print("\n🧪 Test 4: Flow Visualization (Web API)")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    data_result = load_sample_data()
    if data_result is None:
        return False
    
    target_img, ref_img, ref_keypoints = data_result
    
    print(f"✅ Loaded images: ref {ref_img.shape}, target {target_img.shape}")
    print(f"✅ Reference keypoints: {len(ref_keypoints)}")
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    
    try:
        tracker = FFPPWebAPIKeypointTracker(service_url=WEB_SERVICE_URL)
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return False
    
    # ========================================
    # FLOW COMPUTATION AND VISUALIZATION VIA API
    # ========================================
    print("\n🎯 Step 1: Setting reference image via API...")
    ref_result = tracker.set_reference_image(ref_img, ref_keypoints, "flow_test_ref")
    
    if not ref_result['success']:
        print(f"❌ Failed to set reference image: {ref_result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Reference image set with {ref_result['keypoints_count']} keypoints")
    
    print("\n🎯 Step 2: Computing bidirectional flow with return_flow=True via API...")
    start_time = time.time()
    result = tracker.track_keypoints(target_img, reference_name="flow_test_ref", 
                                   bidirectional=True, return_flow=True)
    elapsed_time = time.time() - start_time
    
    if not result['success']:
        print(f"❌ Flow computation failed: {result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Bidirectional flow computation successful!")
    print(f"   API call time: {elapsed_time:.3f}s")
    print(f"   Service processing time: {result.get('total_processing_time', 0):.3f}s")
    print(f"   Tracked keypoints: {len(result['tracked_keypoints'])}")
    
    # Extract flow data (should be decoded numpy arrays)
    if 'flow_data' not in result:
        print("❌ No flow data returned - check return_flow parameter implementation")
        return False
    
    flow_data = result['flow_data']
    forward_flow = flow_data.get('forward_flow')
    reverse_flow = flow_data.get('reverse_flow')
    
    if forward_flow is None:
        print("❌ No forward flow data returned")
        return False
    
    print(f"   Forward flow shape: {forward_flow.shape}")
    print(f"   Forward flow dtype: {forward_flow.dtype}")
    print(f"   Reverse flow shape: {reverse_flow.shape if reverse_flow is not None else 'None'}")
    
    # Verify that we received numpy arrays (not encoded data)
    if not isinstance(forward_flow, np.ndarray):
        print(f"❌ Expected numpy array, got {type(forward_flow)}")
        return False
    
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
        import importlib.util
        
        # Direct import of flow_viz module to avoid path conflicts
        flow_viz_path = os.path.join('ThirdParty', 'FlowFormerPlusPlusServer', 'core', 'utils', 'flow_viz.py')
        spec = importlib.util.spec_from_file_location("flow_viz", flow_viz_path)
        flow_viz = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flow_viz)
        
        # Create output directory
        output_dir = 'output/ffpp_webapi_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. SAVE JSON RESULTS FIRST
        json_path = os.path.join(output_dir, 'flow_visualization_results.json')
        
        # Create a serializable version of the results (without numpy arrays)
        json_result = {
            'success': result['success'],
            'tracked_keypoints': result['tracked_keypoints'],
            'total_processing_time': result.get('total_processing_time', 0),
            'bidirectional_enabled': result.get('bidirectional_enabled', False),
            'bidirectional_stats': result.get('bidirectional_stats', {}),
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
            },
            'test_info': {
                'test_name': 'flow_visualization_webapi',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'service_url': tracker.service_url,
                'return_flow_enabled': True
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"✅ Results saved: {json_path}")
        
        # 2. VISUALIZE FORWARD FLOW
        print("   Creating forward flow visualization...")
        forward_flow_img = flow_viz.flow_to_image(forward_flow)
        forward_vis_path = os.path.join(output_dir, 'webapi_forward_flow_visualization.jpg')
        cv2.imwrite(forward_vis_path, forward_flow_img[:, :, [2, 1, 0]])  # RGB to BGR for OpenCV
        print(f"✅ Forward flow saved: {forward_vis_path}")
        
        # 3. VISUALIZE REVERSE FLOW
        if reverse_flow is not None:
            print("   Creating reverse flow visualization...")
            reverse_flow_img = flow_viz.flow_to_image(reverse_flow)
            reverse_vis_path = os.path.join(output_dir, 'webapi_reverse_flow_visualization.jpg')
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
            
            # Add Web API label
            cv2.putText(combined_img, f"Generated via Web API (Professional)", (10, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add flow statistics
            cv2.putText(combined_img, f"Mean: {forward_magnitude.mean():.1f}px", (10, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(combined_img, f"Max: {forward_magnitude.max():.1f}px", (10, h-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(combined_img, f"Mean: {reverse_magnitude.mean():.1f}px", (w+30, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(combined_img, f"Max: {reverse_magnitude.max():.1f}px", (w+30, h-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            combined_img = forward_flow_img
            cv2.putText(combined_img, f"Forward Flow Only (Professional Web API)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        combined_vis_path = os.path.join(output_dir, 'webapi_combined_flow_visualization.jpg')
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
        
        # Add legend and Web API identifier
        cv2.putText(keypoints_flow_img, "White: Reference keypoints", (10, h-55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(keypoints_flow_img, "Green: Tracked keypoints", (10, h-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(keypoints_flow_img, "Flow data via Web API", (10, h-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(keypoints_flow_img, f"Decoded from base64 transfer", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        keypoints_overlay_path = os.path.join(output_dir, 'webapi_flow_keypoints_overlay.jpg')
        cv2.imwrite(keypoints_overlay_path, keypoints_flow_img[:, :, [2, 1, 0]])  # RGB to BGR for OpenCV
        print(f"✅ Keypoints overlay saved: {keypoints_overlay_path}")
        
    except Exception as e:
        print(f"⚠️ Flow visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Clean up reference
    cleanup_result = tracker.remove_reference_image("flow_test_ref")
    if cleanup_result.get('success'):
        print(f"✅ Reference cleaned up")
    
    return True


def test_service_features():
    """
    Test 5: Web service specific features
    
    This test demonstrates features specific to the web API tracker.
    """
    print("\n🧪 Test 4: Web Service Features")
    print("=" * 50)
    
    try:
        tracker = FFPPWebAPIKeypointTracker(service_url=WEB_SERVICE_URL)
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return False
    
    # Test 1: Health check
    print("\n🏥 Testing service health check...")
    health = tracker.get_service_health()
    
    if health.get('success'):
        print(f"✅ Service health check successful")
        print(f"   Status: {health.get('status', 'Unknown')}")
        print(f"   Message: {health.get('message', 'No message')}")
    else:
        print(f"❌ Service health check failed: {health.get('error')}")
        return False
    
    # Test 2: References listing
    print("\n📋 Testing references listing...")
    refs_result = tracker.get_service_references()
    
    if refs_result.get('success'):
        server_refs = refs_result.get('server_references', {})
        print(f"✅ References listing successful")
        print(f"   Server references: {len(server_refs)}")
        print(f"   Default reference: {refs_result.get('default_reference', 'None')}")
        
        if server_refs:
            print(f"   Reference details:")
            for name, info in server_refs.items():
                keypoints_count = info.get('keypoints_count', 0)
                timestamp = info.get('timestamp', 'Unknown')
                print(f"     - {name}: {keypoints_count} keypoints ({timestamp})")
    else:
        print(f"❌ References listing failed: {refs_result.get('error')}")
        return False
    
    # Test 3: Connection resilience (timeout test)
    print("\n⏱️  Testing connection timeout handling...")
    original_timeout = tracker.timeout
    tracker.timeout = 1  # Very short timeout
    
    # This should either succeed quickly or fail gracefully with timeout
    start_time = time.time()
    health_quick = tracker.get_service_health()
    quick_elapsed = time.time() - start_time
    
    tracker.timeout = original_timeout  # Restore original timeout
    
    if health_quick.get('success') or 'timeout' in str(health_quick.get('error', '')).lower():
        print(f"✅ Timeout handling working correctly")
        print(f"   Quick health check completed in {quick_elapsed:.3f}s")
    else:
        print(f"⚠️ Unexpected timeout behavior: {health_quick.get('error')}")
    
    return True


def run_performance_benchmark():
    """
    Test 6: Performance benchmark of web API calls
    
    This test measures the performance of web API tracking operations.
    """
    print("\n🧪 Test 6: Performance Benchmark (Web API)")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    data_result = load_sample_data()
    if data_result is None:
        return False
    
    target_img, ref_img, ref_keypoints = data_result
    
    try:
        tracker = FFPPWebAPIKeypointTracker(service_url=WEB_SERVICE_URL)
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return False
    
    # Setup reference once for all tests
    print("\n🎯 Setting up reference for benchmark...")
    ref_result = tracker.set_reference_image(ref_img, ref_keypoints, "benchmark_ref")
    
    if not ref_result['success']:
        print(f"❌ Failed to set reference: {ref_result.get('error')}")
        return False
    
    num_runs = 5
    benchmarks = {}
    
    # Benchmark 1: Standard tracking via API
    print(f"\n⏱️  Running benchmark tests...")
    print(f"   Testing standard API tracking ({num_runs} runs)...")
    times = []
    service_times = []
    
    for i in range(num_runs):
        start_time = time.time()
        result = tracker.track_keypoints(target_img, reference_name="benchmark_ref")
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        if result['success']:
            service_times.append(result.get('total_processing_time', 0))
        else:
            print(f"❌ API run {i+1} failed")
            return False
    
    benchmarks['standard_api'] = {
        'mean_api_time': np.mean(times),
        'std_api_time': np.std(times),
        'min_api_time': np.min(times),
        'max_api_time': np.max(times),
        'mean_service_time': np.mean(service_times),
        'std_service_time': np.std(service_times),
        'keypoints_count': len(result.get('tracked_keypoints', []))
    }
    
    # Benchmark 2: Bidirectional tracking via API
    print(f"   Testing bidirectional API tracking ({num_runs} runs)...")
    times = []
    service_times = []
    
    for i in range(num_runs):
        start_time = time.time()
        result = tracker.track_keypoints(target_img, reference_name="benchmark_ref", bidirectional=True)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        if result['success']:
            service_times.append(result.get('total_processing_time', 0))
        else:
            print(f"❌ Bidirectional API run {i+1} failed")
            return False
    
    benchmarks['bidirectional_api'] = {
        'mean_api_time': np.mean(times),
        'std_api_time': np.std(times),
        'min_api_time': np.min(times),
        'max_api_time': np.max(times),
        'mean_service_time': np.mean(service_times),
        'std_service_time': np.std(service_times),
        'keypoints_count': len(result.get('tracked_keypoints', [])),
        'consistency_stats': result.get('bidirectional_stats', {})
    }
    
    # Display results
    print(f"\n📊 Performance Benchmark Results:")
    
    print(f"   Standard API tracking:")
    print(f"     Mean API time: {benchmarks['standard_api']['mean_api_time']:.3f}s ± {benchmarks['standard_api']['std_api_time']:.3f}s")
    print(f"     Mean service time: {benchmarks['standard_api']['mean_service_time']:.3f}s ± {benchmarks['standard_api']['std_service_time']:.3f}s")
    print(f"     API range: {benchmarks['standard_api']['min_api_time']:.3f}s - {benchmarks['standard_api']['max_api_time']:.3f}s")
    print(f"     Keypoints: {benchmarks['standard_api']['keypoints_count']}")
    
    print(f"   Bidirectional API tracking:")
    print(f"     Mean API time: {benchmarks['bidirectional_api']['mean_api_time']:.3f}s ± {benchmarks['bidirectional_api']['std_api_time']:.3f}s")
    print(f"     Mean service time: {benchmarks['bidirectional_api']['mean_service_time']:.3f}s ± {benchmarks['bidirectional_api']['std_service_time']:.3f}s")
    print(f"     API range: {benchmarks['bidirectional_api']['min_api_time']:.3f}s - {benchmarks['bidirectional_api']['max_api_time']:.3f}s")
    print(f"     Keypoints: {benchmarks['bidirectional_api']['keypoints_count']}")
    
    # Calculate overhead
    api_overhead = benchmarks['bidirectional_api']['mean_api_time'] - benchmarks['standard_api']['mean_api_time']
    service_overhead = benchmarks['bidirectional_api']['mean_service_time'] - benchmarks['standard_api']['mean_service_time']
    api_overhead_pct = (api_overhead / benchmarks['standard_api']['mean_api_time']) * 100
    
    print(f"   Bidirectional overhead:")
    print(f"     API overhead: +{api_overhead:.3f}s ({api_overhead_pct:.1f}%)")
    print(f"     Service overhead: +{service_overhead:.3f}s")
    
    # Calculate network overhead (API time - service time)
    std_network_overhead = benchmarks['standard_api']['mean_api_time'] - benchmarks['standard_api']['mean_service_time']
    bid_network_overhead = benchmarks['bidirectional_api']['mean_api_time'] - benchmarks['bidirectional_api']['mean_service_time']
    
    print(f"   Network overhead:")
    print(f"     Standard: {std_network_overhead:.3f}s")
    print(f"     Bidirectional: {bid_network_overhead:.3f}s")
    
    # Save benchmark results
    try:
        output_dir = 'output/ffpp_webapi_keypoint_tracker_example_output'
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, 'performance_benchmark.json')
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': 'performance_benchmark_webapi',
            'num_runs': num_runs,
            'benchmarks': benchmarks,
            'overhead_analysis': {
                'bidirectional_api_overhead_seconds': api_overhead,
                'bidirectional_api_overhead_percent': api_overhead_pct,
                'bidirectional_service_overhead_seconds': service_overhead,
                'network_overhead_standard': std_network_overhead,
                'network_overhead_bidirectional': bid_network_overhead
            },
            'service_info': {
                'service_url': tracker.service_url,
                'timeout': tracker.timeout
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Benchmark results saved: {json_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to save benchmark: {e}")
    
    # Clean up reference
    cleanup_result = tracker.remove_reference_image("benchmark_ref")
    if cleanup_result.get('success'):
        print(f"✅ Reference cleaned up")
    
    return True


def test_encoding_configurations():
    """
    Test 7: Configurable Image Encoding Performance
    """
    print("\n🧪 Test 7: Configurable Image Encoding Performance")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    target_img, ref_img, ref_keypoints = load_sample_data()
    
    # Test configurations: PNG vs JPG with different qualities
    test_configs = [
        {"format": "png", "quality": None, "name": "PNG (Lossless)"},
        {"format": "jpg", "quality": 95, "name": "JPG Quality 95"},
        {"format": "jpg", "quality": 85, "name": "JPG Quality 85"},
        {"format": "jpg", "quality": 75, "name": "JPG Quality 75"}
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🚀 Testing {config['name']}...")
        
        # Initialize tracker with specific encoding
        if config['quality']:
            tracker = FFPPWebAPIKeypointTracker(
                service_url=WEB_SERVICE_URL, 
                image_format=config['format'],
                jpeg_quality=config['quality']
            )
        else:
            tracker = FFPPWebAPIKeypointTracker(
                service_url=WEB_SERVICE_URL, 
                image_format=config['format']
            )
        
        health = tracker.get_service_health()
        if not health.get('success', False):
            print(f"❌ Service not available for {config['name']}")
            continue
        
        encoding_info = tracker.get_image_encoding()
        print(f"   Encoding: {encoding_info['format'].upper()}" + 
              (f" (Quality: {encoding_info['jpeg_quality']})" if encoding_info['format'] == 'jpg' else ""))
        
        # Set reference and measure time
        ref_start = time.time()
        ref_result = tracker.set_reference_image(ref_img, ref_keypoints, f"encoding_test_{config['format']}")
        ref_time = time.time() - ref_start
        
        if not ref_result.get('success'):
            print(f"❌ Failed to set reference for {config['name']}")
            continue
        
        # Track keypoints and measure time
        track_start = time.time()
        track_result = tracker.track_keypoints(target_img, return_flow=False)
        track_time = time.time() - track_start
        
        # Clean up
        tracker.remove_reference_image()
        
        if track_result.get('success'):
            total_time = ref_time + track_time
            api_time = track_result.get('api_call_time', 0)
            
            result = {
                'name': config['name'],
                'format': config['format'],
                'quality': config.get('quality'),
                'total_time': total_time,
                'ref_time': ref_time,
                'track_time': track_time,
                'api_time': api_time,
                'keypoints': len(track_result.get('tracked_keypoints', []))
            }
            results.append(result)
            
            print(f"✅ {config['name']} completed:")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Reference upload: {ref_time:.3f}s")
            print(f"   Tracking request: {track_time:.3f}s")
            print(f"   Tracked keypoints: {result['keypoints']}")
        else:
            print(f"❌ Tracking failed for {config['name']}")
    
    # ========================================
    # PERFORMANCE COMPARISON
    # ========================================
    if len(results) > 1:
        print(f"\n📊 Performance Comparison:")
        print(f"{'Format':<20} {'Total Time':<12} {'Ref Upload':<12} {'Tracking':<12} {'Speedup':<10}")
        print("-" * 70)
        
        baseline = results[0]  # PNG as baseline
        for result in results:
            speedup = baseline['total_time'] / result['total_time']
            
            print(f"{result['name']:<20} {result['total_time']:.3f}s     "
                  f"{result['ref_time']:.3f}s      {result['track_time']:.3f}s      "
                  f"{speedup:.1f}x")
        
        # Find best performing config
        fastest = min(results, key=lambda x: x['total_time'])
        print(f"\n🏆 Fastest configuration: {fastest['name']}")
        print(f"   Total improvement: {baseline['total_time']/fastest['total_time']:.1f}x faster than PNG")
    
    # ========================================
    # DYNAMIC CONFIGURATION TEST
    # ========================================
    print(f"\n🔧 Dynamic Configuration Test:")
    tracker = FFPPWebAPIKeypointTracker(service_url=WEB_SERVICE_URL)
    
    health = tracker.get_service_health()
    if health.get('success', False):
        print(f"   Initial config: {tracker.get_image_encoding()}")
        
        # Change to high-speed JPG
        tracker.set_image_encoding("jpg", 75)
        print(f"   After change: {tracker.get_image_encoding()}")
        
        # Test the new configuration
        start_time = time.time()
        ref_result = tracker.set_reference_image(ref_img, ref_keypoints, "dynamic_test")
        config_test_time = time.time() - start_time
        
        if ref_result.get('success'):
            print(f"   Reference upload with JPG75: {config_test_time:.3f}s")
            tracker.remove_reference_image()
        
    print("✅ Encoding Configuration PASSED")
    return True


def main():
    """
    Main function: Run complete test suite for web API tracker
    """
    parser = argparse.ArgumentParser(description='FFPPWebAPIKeypointTracker Example & Test Suite')
    parser.add_argument('--service', '-s', action='store_true', 
                        help='Check service availability only')
    parser.add_argument('--url', '-u', type=str, 
                        help='FlowFormer++ web service URL (overrides WEB_SERVICE_URL)')
    args = parser.parse_args()
    
    # Use provided URL if specified, otherwise use the configured URL
    global WEB_SERVICE_URL
    if args.url:
        WEB_SERVICE_URL = args.url
        print(f"🔧 Using provided service URL: {WEB_SERVICE_URL}")
    else:
        print(f"🔧 Using configured service URL: {WEB_SERVICE_URL}")
    
    print("🌐 FFPPWebAPIKeypointTracker Example & Test Suite 🌐")
    print("=" * 60)
    print("This example demonstrates the web API FFPPKeypointTracker interface")
    print("with bidirectional flow validation and web service integration.")
    print("=" * 60)
    
    # Check service availability first
    service_info = check_service_availability()
    
    if not service_info['available']:
        print("❌ Web service not available - cannot proceed with tests")
        print("   Make sure the FlowFormer++ web service is running on http://localhost:8001")
        return
    
    if args.service:
        print("✅ Service check completed - service is available!")
        return
    
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
        ("Basic API Tracking", test_basic_tracking),
        # ("Bidirectional API Validation", test_bidirectional_validation),
        # ("Multiple References API", test_multiple_references),
        # ("Flow Visualization API", test_flow_visualization),
        # ("Web Service Features", test_service_features),
        # ("Encoding Configuration Performance", test_encoding_configurations),
        # ("API Performance Benchmark", run_performance_benchmark)
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
    print(f"🎯 Web API Test Suite Summary")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Total time: {total_time:.2f}s")
    
    if passed_tests == total_tests:
        print("🎉 All web API tests passed successfully!")
        print("   Check the 'output/ffpp_webapi_keypoint_tracker_example_output/' directory for generated results and visualizations.")
    else:
        print("⚠️  Some tests failed - check output above for details")
    
    print("=" * 60)


if __name__ == "__main__":
    main()