#!/usr/bin/env python3
"""
Simple Keypoint Tracker Demo
============================

This script demonstrates basic usage of the core.KeypointTracker class
to perform keypoint tracking with optical flow using sample data.

Uses images from sample_data/flow_image_pair/ and outputs to output/.
"""

import os
import sys

# Add project root to path to import core module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import from core module
from core import KeypointTracker


def display_visualization(image_path):
    """Display the visualization image using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use interactive backend for display
        import matplotlib.pyplot as plt
        from PIL import Image
        
        # Load and display the image
        img = Image.open(image_path)
        plt.figure(figsize=(15, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Keypoint Tracking Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    except ImportError as e:
        print(f"⚠️ Could not display image: {e}")
        print(f"📁 Please manually open: {image_path}")
    except Exception as e:
        print(f"⚠️ Error displaying image: {e}")
        print(f"📁 Please manually open: {image_path}")


def main():
    """Main function - simple keypoint tracking demo."""
    print("🎯 Simple Keypoint Tracking Demo")
    print("=" * 40)
    
    try:
        # Create tracker instance
        print("🔧 Initializing KeypointTracker...")
        tracker = KeypointTracker(server_url="http://msraig-ubuntu-3:5000")
        
        # Run tracking with default settings
        print("🚀 Running keypoint tracking...")
        result = tracker.run_tracking()
        
        # Check results
        if result['success']:
            print("\n✅ Tracking completed successfully!")
            print(f"📁 Results saved:")
            print(f"   • Visualization: {result['output_paths']['visualization']}")
            print(f"   • Data: {result['output_paths']['data']}")
            print(f"⏱️ Flow computation time: {result['flow_time']:.2f} seconds")
            
            # Show keypoint movements
            print(f"\n📊 Tracked {len(result['tracked_keypoints'])} keypoints:")
            for kp in result['tracked_keypoints']:
                import numpy as np
                movement = np.sqrt(kp['flow_x']**2 + kp['flow_y']**2)
                print(f"   • {kp['name']}: moved {movement:.1f} pixels")
            
            # Display the visualization
            print("\n🖼️ Displaying visualization...")
            display_visualization(result['output_paths']['visualization'])
                
        else:
            print(f"❌ Tracking failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
