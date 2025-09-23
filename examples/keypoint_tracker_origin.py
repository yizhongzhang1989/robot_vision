#!/usr/bin/env python3
"""
Keypoint Tracking with Optical Flow

This script loads keypoints from sample_data/flow_image_pair/ref_img_keypoints.json, computes optical flow
between sample_data/flow_image_pair/ref_img.jpg and sample_data/flow_image_pair/comp_img.jpg, and tracks where the keypoints move
to in the comparison image. Results are saved to output/.

Paths are automatically determined relative to the script location.
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ThirdParty', 'FlowFormerPlusPlusServer'))
from flowformer_api import FlowFormerClient

# Global paths - automatically determine based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, 'sample_data', 'flow_image_pair')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


def load_keypoints(json_path):
    """Load keypoints from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    keypoints = []
    for kp in data['keypoints']:
        keypoints.append({
            'id': kp['id'],
            'name': kp['name'],
            'x': kp['x'],
            'y': kp['y']
        })
    
    original_size = (data['image']['width'], data['image']['height'])
    return keypoints, original_size


def resize_keypoints(keypoints, original_size, new_size):
    """Resize keypoints coordinates to match resized image."""
    orig_w, orig_h = original_size
    new_w, new_h = new_size
    
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    
    resized_keypoints = []
    for kp in keypoints:
        resized_kp = kp.copy()
        resized_kp['x'] = kp['x'] * scale_x
        resized_kp['y'] = kp['y'] * scale_y
        resized_keypoints.append(resized_kp)
    
    return resized_keypoints


def track_keypoints_with_flow(keypoints, flow):
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


def visualize_keypoint_tracking(ref_img, comp_img, original_keypoints, tracked_keypoints):
    """Create visualization showing keypoint tracking results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display reference image with original keypoints
    ax1.imshow(ref_img)
    ax1.set_title('Reference Image - Original Keypoints', fontsize=14, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']
    
    for i, kp in enumerate(original_keypoints):
        color = colors[i % len(colors)]
        ax1.plot(kp['x'], kp['y'], 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
    
    ax1.set_xlim(0, ref_img.shape[1])
    ax1.set_ylim(ref_img.shape[0], 0)
    ax1.axis('off')
    
    # Display comparison image with tracked keypoints
    ax2.imshow(comp_img)
    ax2.set_title('Comparison Image - Tracked Keypoints', fontsize=14, fontweight='bold')
    
    for i, kp in enumerate(tracked_keypoints):
        color = colors[i % len(colors)]
        # Plot tracked position
        ax2.plot(kp['new_x'], kp['new_y'], 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # Draw arrow showing movement
        ax2.arrow(kp['x'], kp['y'], kp['flow_x'], kp['flow_y'], 
                 head_width=10, head_length=15, fc=color, ec=color, alpha=0.6, linewidth=2)
    
    ax2.set_xlim(0, comp_img.shape[1])
    ax2.set_ylim(comp_img.shape[0], 0)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to numpy array for saving
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    buf = np.asarray(buf)
    buf = buf[:, :, :3]  # Convert RGBA to RGB
    
    plt.close(fig)
    return buf


def display_tracking_results():
    """Display the keypoint tracking results using matplotlib."""
    
    # Load the visualization image
    try:
        # Switch to interactive backend for display
        matplotlib.use('TkAgg')
        
        img = Image.open(os.path.join(OUTPUT_DIR, "keypoint_tracking.png"))
        
        # Create matplotlib figure
        plt.figure(figsize=(16, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Keypoint Tracking Results', fontsize=16, fontweight='bold', pad=20)
        
        # Load and display tracking data
        with open(os.path.join(OUTPUT_DIR, "tracked_keypoints.json"), 'r') as f:
            data = json.load(f)
        
        # Print tracking summary
        print("\nKeypoint Tracking Summary:")
        print("=" * 50)
        
        for kp in data['tracked_keypoints']:
            movement_distance = (kp['flow_x']**2 + kp['flow_y']**2)**0.5
            print(f"\n{kp['name']}:")
            print(f"  Original position: ({kp['x']:.1f}, {kp['y']:.1f})")
            print(f"  Tracked position:  ({kp['new_x']:.1f}, {kp['new_y']:.1f})")
            print(f"  Movement vector:   ({kp['flow_x']:.1f}, {kp['flow_y']:.1f})")
            print(f"  Movement distance: {movement_distance:.1f} pixels")
        
        print(f"\nFlow computation time: {data['flow_computation_time']:.2f} seconds")
        print(f"Image dimensions: {data['image_dimensions']['reference']}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        print("\n✅ Keypoint tracking visualization displayed!")
        
        # Switch back to non-interactive backend
        matplotlib.use('Agg')
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files.")
        print(f"Missing file: {e.filename}")
    except Exception as e:
        print(f"❌ Error displaying results: {e}")


def main():
    print("Keypoint Tracking with Optical Flow")
    print("===================================")
    print()
    
    # Initialize client with remote server
    server_url = "http://msraig-ubuntu-3:5000"
    client = FlowFormerClient(server_url=server_url, timeout=180)
    
    # Test server connection
    print("🔗 Connecting to remote server...")
    if not client.setup():
        print("❌ Failed to connect to remote server")
        return False
    print("✅ Successfully connected to remote server!")
    print()
    
    # Check for required files
    ref_img_path = os.path.join(SAMPLE_DATA_DIR, "ref_img.jpg")
    comp_img_path = os.path.join(SAMPLE_DATA_DIR, "comp_img.jpg")
    keypoints_path = os.path.join(SAMPLE_DATA_DIR, "ref_img_keypoints.json")
    
    missing_files = []
    for file_path in [ref_img_path, comp_img_path, keypoints_path]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    try:
        # Load keypoints
        print("📍 Loading keypoints...")
        keypoints, original_size = load_keypoints(keypoints_path)
        print(f"   Loaded {len(keypoints)} keypoints from {keypoints_path}")
        print(f"   Original image size: {original_size}")
        for kp in keypoints:
            print(f"   - {kp['name']}: ({kp['x']:.1f}, {kp['y']:.1f})")
        print()
        
        # Load and resize images
        print("📷 Loading and resizing images...")
        ref_img_pil = Image.open(ref_img_path)
        comp_img_pil = Image.open(comp_img_path)
        
        # Resize to width = 800
        target_width = 800
        
        # Resize reference image
        orig_w1, orig_h1 = ref_img_pil.size
        aspect_ratio1 = orig_h1 / orig_w1
        target_height1 = int(target_width * aspect_ratio1)
        ref_img_resized = ref_img_pil.resize((target_width, target_height1), Image.Resampling.LANCZOS)
        
        # Resize comparison image
        orig_w2, orig_h2 = comp_img_pil.size
        aspect_ratio2 = orig_h2 / orig_w2
        target_height2 = int(target_width * aspect_ratio2)
        comp_img_resized = comp_img_pil.resize((target_width, target_height2), Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        ref_img = np.array(ref_img_resized)
        comp_img = np.array(comp_img_resized)
        
        print(f"   Reference image: {orig_w1}x{orig_h1} -> {ref_img.shape[1]}x{ref_img.shape[0]}")
        print(f"   Comparison image: {orig_w2}x{orig_h2} -> {comp_img.shape[1]}x{comp_img.shape[0]}")
        print()
        
        # Resize keypoints to match resized images
        resized_keypoints = resize_keypoints(keypoints, original_size, (target_width, target_height1))
        print("📍 Resized keypoints coordinates:")
        for kp in resized_keypoints:
            print(f"   - {kp['name']}: ({kp['x']:.1f}, {kp['y']:.1f})")
        print()
        
        # Compute optical flow
        print("🚀 Computing optical flow...")
        start_time = time.time()
        flow = client.compute_flow(ref_img, comp_img)
        flow_time = time.time() - start_time
        print(f"✅ Flow computed in {flow_time:.2f}s")
        print(f"   Flow shape: {flow.shape}")
        print(f"   Flow range: [{flow.min():.2f}, {flow.max():.2f}]")
        print()
        
        # Track keypoints using flow
        print("🎯 Tracking keypoints...")
        tracked_keypoints = track_keypoints_with_flow(resized_keypoints, flow)
        
        print("📊 Keypoint tracking results:")
        for kp in tracked_keypoints:
            movement = np.sqrt(kp['flow_x']**2 + kp['flow_y']**2)
            print(f"   - {kp['name']}:")
            print(f"     Original: ({kp['x']:.1f}, {kp['y']:.1f})")
            print(f"     Tracked:  ({kp['new_x']:.1f}, {kp['new_y']:.1f})")
            print(f"     Movement: ({kp['flow_x']:.1f}, {kp['flow_y']:.1f}) pixels")
            print(f"     Distance: {movement:.1f} pixels")
        print()
        
        # Create visualization
        print("🎨 Creating keypoint tracking visualization...")
        tracking_vis = visualize_keypoint_tracking(ref_img, comp_img, resized_keypoints, tracked_keypoints)
        
        # Save results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save visualization
        vis_output_path = os.path.join(OUTPUT_DIR, "keypoint_tracking.png")
        Image.fromarray(tracking_vis).save(vis_output_path)
        print(f"✅ Keypoint tracking visualization saved to: {vis_output_path}")
        
        # Save tracked keypoints as JSON
        tracked_output_path = os.path.join(OUTPUT_DIR, "tracked_keypoints.json")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        output_data = {
            "original_keypoints": convert_numpy_types(resized_keypoints),
            "tracked_keypoints": convert_numpy_types(tracked_keypoints),
            "flow_computation_time": float(flow_time),
            "image_dimensions": {
                "reference": list(ref_img.shape),
                "comparison": list(comp_img.shape)
            }
        }
        
        with open(tracked_output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✅ Tracked keypoints data saved to: {tracked_output_path}")
        
        print()
        print("🎉 Keypoint tracking completed successfully!")
        
        # Display the results
        print("\n🖼️  Displaying tracking results...")
        display_tracking_results()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during keypoint tracking: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n💡 Results Summary:")
        print("   - Keypoints loaded from ref_img_keypoints.json")
        print("   - Optical flow computed between reference and comparison images")
        print("   - Keypoint positions tracked using flow vectors")
        print("   - Visualization saved showing original and tracked positions")
        print("   - Movement arrows show keypoint displacement")
        print("   - Interactive display window opened showing results")
    else:
        print("\n🔧 Troubleshooting:")
        print(f"   - Ensure all required files exist in {SAMPLE_DATA_DIR} (ref_img.jpg, comp_img.jpg, ref_img_keypoints.json)")
        print("   - Check server connectivity")
        print("   - Verify image formats are supported")
        print("\n📺 To display results manually, run:")
        print("   python -c \"from keypoint_tracker import display_tracking_results; display_tracking_results()\")")