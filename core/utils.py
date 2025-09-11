"""
Utility functions for robot vision tasks
========================================

Common utility functions used across different vision modules.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image


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


def visualize_tracking_results(ref_img, comp_img, original_keypoints, tracked_keypoints):
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


def get_project_paths():
    """Get standard project paths."""
    # Get path relative to this file location  
    core_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(core_dir)
    
    paths = {
        'project_root': project_root,
        'core_dir': core_dir,
        'test_data': os.path.join(project_root, 'test_data'),
        'output': os.path.join(project_root, 'test_data', 'output'),
        'thirdparty': os.path.join(project_root, 'ThirdParty', 'FlowFormerPlusPlusServer')
    }
    
    return paths
