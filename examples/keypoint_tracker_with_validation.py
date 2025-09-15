#!/usr/bin/env python3
"""
Test script for keypoint tracking with reverse validation
========================================================

This script tests the new reverse validation functionality added to KeypointTracker.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.keypoint_tracker import KeypointTracker
from core.utils import load_keypoints, resize_keypoints

def main():
    """Keypoint tracking with reverse validation."""
    print("🧪 Start Keypoint Tracking with Reverse Validation")
    print("=" * 60)
    
    try:
        # Initialize tracker
        print("\n🚀 Initializing KeypointTracker...")
        tracker = KeypointTracker()
        
        # Step 1: Normal keypoint tracking
        tracking_result = tracker.run_tracking(verbose=True)
        
        if not tracking_result['success']:
            print("\n❌ Tracking failed!")
            print(f"   Error: {tracking_result.get('error', 'Unknown')}")
            return False
        
        # Step 2: Load images and original keypoints for reverse validation
        print("📊 Preparing for reverse validation...")
        
        # Load images
        ref_img_path = os.path.join(tracker.paths['test_data'], 'ref_img.jpg')
        comp_img_path = os.path.join(tracker.paths['test_data'], 'comp_img.jpg')
        ref_img, comp_img = tracker.load_images(ref_img_path=ref_img_path, comp_img_path=comp_img_path)

        # Load and resize original keypoints
        keypoints_json_path = os.path.join(tracker.paths['test_data'], 'ref_img_knobs.json')
        original_keypoints, original_size = load_keypoints(keypoints_json_path)
        resized_original_keypoints = resize_keypoints(original_keypoints, original_size, ref_img.shape[:2][::-1])
        
        # Step 3: Run reverse validation
        print("\n🔍 Running reverse validation...")
        validation_result = tracker.reverse_validation(
            tracking_result['tracked_keypoints'], 
            resized_original_keypoints,
            ref_img, 
            comp_img, 
            verbose=True
        )
        
        if validation_result['success']:
            print("\n✅ Complete workflow finished successfully!")
            
            # Print summary of validation results
            val_summary = validation_result['validation_results']
            print(f"\n📈 Validation Summary:")
            print(f"   • Total keypoints validated: {val_summary['total_keypoints']}")
            print(f"   • Average error: {val_summary['average_error']:.2f} pixels")
            print(f"   • Maximum error: {val_summary['max_error']:.2f} pixels")
            print(f"   • Minimum error: {val_summary['min_error']:.2f} pixels")
            print(f"   • Error threshold: {val_summary['error_threshold']:.1f} pixels")
            
            validation_status = "PASSED ✅" if val_summary['validation_passed'] else "FAILED ❌"
            print(f"   • Validation result: {validation_status}")
            
            # Print file paths
            val_paths = validation_result['output_paths']
            track_paths = tracking_result['output_paths']
            print(f"\n📁 Output files:")
            print(f"   • Validation visualization: {val_paths['visualization']}")
            print(f"   • Validation data: {val_paths['validation_data']}")
            print(f"   • Tracking visualization: {track_paths['visualization']}")
            print(f"   • Tracking data: {track_paths['data']}")
        else:
            print("\n❌ Validation failed!")
            print(f"   Error: {validation_result.get('error', 'Unknown')}")
        
        return tracking_result['success'] and validation_result['success']
        
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("📋 Check the output directory for validation results and visualizations.")
        sys.exit(0)
    else:
        print("\n💀 Test failed!")
        sys.exit(1)