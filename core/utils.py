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


def compare_keypoints(original_keypoints, reverse_tracked_keypoints, error_threshold=2.0):
    """Compare original keypoints with reverse-tracked keypoints.
    
    Args:
        original_keypoints: Original keypoints from ref_img
        reverse_tracked_keypoints: Keypoints tracked back from comp_img
        error_threshold: Maximum acceptable error in pixels
        
    Returns:
        dict: Comparison results with errors and metrics
    """
    individual_results = []
    errors = []
    
    for orig_kp, rev_kp in zip(original_keypoints, reverse_tracked_keypoints):
        # Calculate Euclidean distance between original and reverse-tracked positions
        orig_pos = np.array([orig_kp['x'], orig_kp['y']])
        rev_pos = np.array([rev_kp['new_x'], rev_kp['new_y']])
        
        distance = np.linalg.norm(orig_pos - rev_pos)
        errors.append(distance)
        
        result = {
            'keypoint_name': orig_kp.get('name', f"keypoint_{orig_kp.get('id', 'unknown')}"),
            'original_position': {'x': float(orig_kp['x']), 'y': float(orig_kp['y'])},
            'reverse_tracked_position': {'x': float(rev_kp['new_x']), 'y': float(rev_kp['new_y'])},
            'error_distance_pixels': float(distance),
            'error_within_threshold': bool(distance <= error_threshold),
            'flow_vector': {'x': float(rev_kp['flow_x']), 'y': float(rev_kp['flow_y'])}
        }
        individual_results.append(result)
    
    # Calculate overall metrics
    average_error = float(np.mean(errors)) if errors else 0.0
    max_error = float(np.max(errors)) if errors else 0.0
    min_error = float(np.min(errors)) if errors else 0.0
    validation_passed = all(error <= error_threshold for error in errors)
    
    return {
        'individual_results': individual_results,
        'average_error': average_error,
        'max_error': max_error,
        'min_error': min_error,
        'validation_passed': bool(validation_passed),
        'error_threshold': error_threshold,
        'total_keypoints': len(original_keypoints)
    }


def visualize_reverse_validation_results(ref_img, comp_img, original_keypoints, 
                                        forward_tracked_keypoints, reverse_tracked_keypoints, 
                                        comparison_results):
    """Create visualization showing reverse validation results.
    
    Args:
        ref_img: Reference image
        comp_img: Comparison image  
        original_keypoints: Original keypoints from ref_img
        forward_tracked_keypoints: Forward tracking results
        reverse_tracked_keypoints: Reverse tracking results
        comparison_results: Comparison metrics from compare_keypoints()
        
    Returns:
        numpy.ndarray: Validation visualization image
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: Original keypoints on ref_img
    axes[0, 0].imshow(ref_img)
    axes[0, 0].set_title('Original Keypoints (Reference Image)', fontweight='bold')
    for kp in original_keypoints:
        axes[0, 0].plot(kp['x'], kp['y'], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
        axes[0, 0].annotate(kp.get('name', f"kp_{kp.get('id', '')}"), 
                          (kp['x'], kp['y']), xytext=(5, 5), textcoords='offset points',
                          color='white', fontweight='bold', fontsize=8,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    axes[0, 0].axis('off')
    
    # Top-right: Forward tracked keypoints on comp_img
    axes[0, 1].imshow(comp_img)
    axes[0, 1].set_title('Forward Tracked Keypoints (Comparison Image)', fontweight='bold')
    for kp in forward_tracked_keypoints:
        axes[0, 1].plot(kp['new_x'], kp['new_y'], 'go', markersize=8, markeredgecolor='white', markeredgewidth=2)
        axes[0, 1].annotate(kp.get('name', f"kp_{kp.get('id', '')}"), 
                          (kp['new_x'], kp['new_y']), xytext=(5, 5), textcoords='offset points',
                          color='white', fontweight='bold', fontsize=8,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))
    axes[0, 1].axis('off')
    
    # Bottom-left: Comparison - original vs reverse tracked on ref_img
    axes[1, 0].imshow(ref_img)
    axes[1, 0].set_title('Validation: Original vs Reverse Tracked', fontweight='bold')
    
    for orig_kp, rev_kp, result in zip(original_keypoints, reverse_tracked_keypoints, comparison_results['individual_results']):
        # Original keypoints in red
        axes[1, 0].plot(orig_kp['x'], orig_kp['y'], 'ro', markersize=8, 
                      markeredgecolor='white', markeredgewidth=2, 
                      label='Original' if orig_kp == original_keypoints[0] else "")
        
        # Reverse tracked keypoints in blue
        axes[1, 0].plot(rev_kp['new_x'], rev_kp['new_y'], 'bo', markersize=8, 
                      markeredgecolor='white', markeredgewidth=2, 
                      label='Reverse Tracked' if rev_kp == reverse_tracked_keypoints[0] else "")
        
        # Draw connecting line with color based on error
        color = 'green' if result['error_within_threshold'] else 'red'
        line_style = '-' if result['error_within_threshold'] else '--'
        axes[1, 0].plot([orig_kp['x'], rev_kp['new_x']], [orig_kp['y'], rev_kp['new_y']], 
                      color=color, linestyle=line_style, linewidth=2, alpha=0.7)
        
        # Add error distance annotation
        mid_x = (orig_kp['x'] + rev_kp['new_x']) / 2
        mid_y = (orig_kp['y'] + rev_kp['new_y']) / 2
        axes[1, 0].annotate(f"{result['error_distance_pixels']:.1f}px", 
                          (mid_x, mid_y), xytext=(0, -10), textcoords='offset points',
                          color='black', fontweight='bold', fontsize=7, ha='center',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].axis('off')
    
    # Bottom-right: Error statistics
    axes[1, 1].axis('off')
    stats_text = f"""Validation Statistics:

Total Keypoints: {comparison_results['total_keypoints']}
Average Error: {comparison_results['average_error']:.2f} pixels
Maximum Error: {comparison_results['max_error']:.2f} pixels  
Minimum Error: {comparison_results['min_error']:.2f} pixels
Error Threshold: {comparison_results['error_threshold']:.1f} pixels

Validation Result: {'PASSED' if comparison_results['validation_passed'] else 'FAILED'}

Keypoint Details:"""
    
    for result in comparison_results['individual_results']:
        status = "✓" if result['error_within_threshold'] else "✗"
        stats_text += f"\n{status} {result['keypoint_name']}: {result['error_distance_pixels']:.2f}px"
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert plot to numpy array
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
