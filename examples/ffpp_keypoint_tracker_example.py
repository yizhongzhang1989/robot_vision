#!/usr/bin/env python3
"""
FFPPKeypointTracker Example and Test Suite

This example demonstrates the usage of FFPPKeypointTracker with the simplified interface
and bidirectional flow validation features. It serves as both a usage example and
automated test suite.

Code Structure:
- load_sample_data(): Centralized data loading (images + keypoints)
- test_basic_tracking(): Demonstrates core two-step tracking process
- test_bidirectional_validation(): Shows accuracy assessment features
- test_multiple_references(): Multiple reference management demo
- run_performance_benchmark(): Performance comparison between modes

Features demonstrated:
- Basic keypoint tracking with stored references
- Multiple reference image management  
- Bidirectional flow validation for accuracy assessment
- Output visualization and JSON export
- Performance benchmarking

Usage:
    python examples/ffpp_keypoint_tracker_example.py
"""

import sys
import os
import cv2
import json
import numpy as np
import time

# Add the parent directory to the path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.ffpp_keypoint_tracker import FFPPKeypointTracker


def load_sample_data():
    """
    Load sample data for testing: reference image, target image, and keypoints.
    
    Returns:
        dict: Dictionary containing:
            - 'ref_img': Reference image (numpy array)
            - 'target_img': Target/comparison image (numpy array) 
            - 'keypoints': List of keypoints from JSON file
            - 'success': Boolean indicating if data was loaded successfully
            - 'error': Error message if loading failed
    """
    print("📁 Loading sample data...")
    
    # Define paths
    ref_image_path = 'sample_data/flow_image_pair/ref_img.jpg'
    comp_image_path = 'sample_data/flow_image_pair/comp_img.jpg'
    ref_keypoints_path = 'sample_data/flow_image_pair/ref_img_keypoints.json'
    
    try:
        # Load images
        ref_img = cv2.imread(ref_image_path)
        target_img = cv2.imread(comp_image_path)
        
        if ref_img is None or target_img is None:
            return {
                'success': False,
                'error': 'Failed to load sample images - check if files exist'
            }
        
        # Load keypoints
        with open(ref_keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        
        keypoints = keypoints_data['keypoints']
        
        print(f"✅ Sample data loaded successfully:")
        print(f"   Reference image: {ref_img.shape}")
        print(f"   Target image: {target_img.shape}")  
        print(f"   Keypoints: {len(keypoints)}")
        
        return {
            'success': True,
            'ref_img': ref_img,
            'target_img': target_img,
            'keypoints': keypoints
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error loading sample data: {e}'
        }


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
    data = load_sample_data()
    if not data['success']:
        print(f"❌ {data['error']}")
        return False
    
    ref_img = data['ref_img']
    target_img = data['target_img']
    test_keypoints = data['keypoints']
    
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
    ref_result = tracker.set_reference_image(ref_img, test_keypoints)
    
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
    
    # Show displacement statistics
    if tracked_count > 0:
        displacements = [(kp.get('displacement_x', 0), kp.get('displacement_y', 0)) 
                        for kp in result['tracked_keypoints']]
        avg_displacement = np.mean([np.sqrt(dx**2 + dy**2) for dx, dy in displacements])
        print(f"   Average displacement: {avg_displacement:.1f} pixels")
    
    # ========================================
    # OUTPUT AND VISUALIZATION
    # ========================================
    print("\n📊 Creating outputs...")
    try:
        # Create output directory
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        vis_img = target_img.copy()
        for i, kp in enumerate(result['tracked_keypoints']):
            x, y = int(round(kp['x'])), int(round(kp['y']))
            h, w = vis_img.shape[:2]
            
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(vis_img, str(i+1), (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save outputs
        vis_path = os.path.join(output_dir, 'basic_tracking_visualization.jpg')
        cv2.imwrite(vis_path, vis_img)
        
        json_path = os.path.join(output_dir, 'basic_tracking_results.json')
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': 'basic_tracking',
            'initialization_time': init_elapsed_time,
            'tracking_time': elapsed_time,
            'keypoints_count': tracked_count,
            'tracked_keypoints': result['tracked_keypoints'],
            'processing_stats': result.get('processing_stats', {})
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Visualization saved: {vis_path}")
        print(f"✅ Results saved: {json_path}")
        
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
    data = load_sample_data()
    if not data['success']:
        print(f"❌ {data['error']}")
        return False
    
    ref_img = data['ref_img']
    target_img = data['target_img']
    test_keypoints = data['keypoints']
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    tracker = FFPPKeypointTracker()
    
    if not tracker.model_loaded:
        print("❌ Failed to load model")
        return False
    
    # Set reference and track with bidirectional validation
    print("\n🎯 Testing bidirectional flow validation...")
    tracker.set_reference_image(ref_img, test_keypoints)
    
    start_time = time.time()
    result = tracker.track_keypoints(target_img, bidirectional=True)
    elapsed_time = time.time() - start_time
    
    if not result['success']:
        print(f"❌ Bidirectional tracking failed: {result.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Bidirectional tracking completed in {elapsed_time:.3f}s")
    
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
    
    # Save bidirectional results
    print("\n💾 Saving bidirectional results...")
    try:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, 'bidirectional_validation_results.json')
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': 'bidirectional_validation',
            'tracking_time': elapsed_time,
            'keypoints_count': len(result.get('tracked_keypoints', [])),
            'bidirectional_stats': result.get('bidirectional_stats', {}),
            'tracked_keypoints': result['tracked_keypoints']
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Bidirectional results saved: {json_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
    
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
    data = load_sample_data()
    if not data['success']:
        print(f"❌ {data['error']}")
        return False
    
    ref_image = data['ref_img']
    comp_image = data['target_img']
    
    # Use the loaded keypoints directly for different reference scenarios
    all_keypoints = data['keypoints']
    half_keypoints = all_keypoints[:len(all_keypoints)//2]  # First half for testing
    
    print(f"✅ Using keypoint sets from loaded data:")
    print(f"   Full set: {len(all_keypoints)} keypoints")
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
    tracker.set_reference_image(ref_image, all_keypoints)
    print(f"   Set default reference: {len(all_keypoints)} keypoints")
    
    # Reference 2: Half keypoint set
    tracker.set_reference_image(ref_image, half_keypoints, image_name="half_set")
    print(f"   Set 'half_set' reference: {len(half_keypoints)} keypoints")
    
    # Reference 3: Image only (no keypoints)
    tracker.set_reference_image(ref_image, image_name="image_only")
    print(f"   Set 'image_only' reference: 0 keypoints")
    
    print(f"\n📍 Active references: {list(tracker.reference_data.keys())}")
    print(f"   Default reference: '{tracker.default_reference_key}'")
    
    # Test tracking with different references
    print("\n🎯 Testing tracking with different references...")
    
    results = {}
    
    # Track with default reference
    start_time = time.time()
    result_default = tracker.track_keypoints(comp_image)
    elapsed_time = time.time() - start_time
    results['default'] = result_default
    print(f"   Default: {elapsed_time:.3f}s - {len(result_default.get('tracked_keypoints', []))} points")
    
    # Track with specific references
    for ref_name in ["half_set", "image_only"]:
        start_time = time.time()
        result = tracker.track_keypoints(comp_image, reference_name=ref_name)
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
        output_dir = 'output'
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


def run_performance_benchmark():
    """
    Performance benchmark: Measure and compare different tracking modes
    """
    print("\n🧪 Performance Benchmark")
    print("=" * 50)
    
    # ========================================
    # DATA PREPARATION
    # ========================================
    data = load_sample_data()
    if not data['success']:
        print(f"❌ {data['error']}")
        return False
    
    ref_img = data['ref_img']
    target_img = data['target_img']
    test_keypoints = data['keypoints']
    
    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing tracker...")
    tracker = FFPPKeypointTracker()
    
    if not tracker.model_loaded:
        print("❌ Failed to load model")
        return False
    
    # Set reference once
    tracker.set_reference_image(ref_img, test_keypoints)
    
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
        output_dir = 'output'
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
        print("   Check the 'output/' directory for generated results and visualizations.")
    else:
        print("⚠️  Some tests failed - check output above for details")
    
    print("=" * 60)


if __name__ == "__main__":
    main()