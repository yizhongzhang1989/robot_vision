#!/usr/bin/env python3
"""
Test Dataset Runner

This script automatically tests keypoint tracking across all subdirectories in /test_dataset.

================================================================================
TEST DATASET ORGANIZATION
================================================================================

Directory Structure:
--------------------
test_dataset/
├── subdir_1/                    # Test scenario 1
│   ├── image1.jpg              # Reference image (has JSON)
│   ├── image1.json             # Keypoint annotations for image1
│   ├── image2.jpg              # Target image (may or may not have JSON)
│   ├── image3.jpg              # Another target image
│   └── ...
├── subdir_2/                    # Test scenario 2
│   ├── ref_image.jpg
│   ├── ref_image.json
│   ├── target1.jpg
│   └── ...
└── reports/                     # Auto-generated reports (excluded from testing)
    ├── subdir_1/
    │   ├── report.html
    │   ├── image1_to_image2.jpg
    │   └── ...
    └── subdir_2/
        └── ...

JSON Keypoint Format:
---------------------
Each JSON file should contain keypoint annotations in one of these formats:

Format 1 (with keypoints key):
{
    "keypoints": [
        {"x": 100.5, "y": 200.3},
        {"x": 150.2, "y": 180.7},
        ...
    ]
}

Format 2 (direct array):
[
    {"x": 100.5, "y": 200.3},
    {"x": 150.2, "y": 180.7},
    ...
]

Testing Process:
----------------
For each subdirectory:
1. Find all image-JSON pairs (reference images with keypoint annotations)
2. For each reference image:
   - Set it as the reference with its keypoints
   - Track keypoints on ALL other images in the same subdirectory
   - This includes images that have JSON and images that don't
3. Generate visualization with color-coded error
4. Create HTML report with statistics

Example Test Scenario:
----------------------
If subdir_1 contains:
- image1.jpg + image1.json (5 keypoints)
- image2.jpg + image2.json (3 keypoints)  
- image3.jpg (no JSON)

The script will perform:
1. Reference: image1.jpg → Track on image2.jpg and image3.jpg
2. Reference: image2.jpg → Track on image1.jpg and image3.jpg
Total: 4 tracking tests

Output Reports:
---------------
For each subdirectory, generates:
- HTML report: test_dataset/reports/<subdir_name>/report.html
- Visualization images: Side-by-side comparisons with color-coded keypoints
  - Left: Reference image with red crosses
  - Right: Target image with color-coded crosses (green=good, red=bad)

================================================================================

Script Functionality:
1. Find all images with corresponding JSON keypoint files
2. For each reference image (with JSON), track keypoints on all other images in the directory
3. Track both images with and without JSON files
4. Always uses bidirectional validation for accuracy assessment
5. Generates color-coded visualization based on tracking error

Color Coding:
- Reference keypoints: Red crosses
- Tracked keypoints: Color-coded from green (low error) to red (high error)
- Default max error threshold: 10.0 pixels (configurable)

The script uses the FlowFormer++ web API for tracking, similar to the example in
examples/ffpp_webapi_keypoint_tracker_example.py.

Usage:
    python tests/run_test_dataset.py
    python tests/run_test_dataset.py --url http://server:8001
    python tests/run_test_dataset.py --max-error 15.0
"""

import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.ffpp_webapi_keypoint_tracker import FFPPWebAPIKeypointTracker

# =============================================================================
# CONFIGURATION
# =============================================================================
WEB_SERVICE_URL = "http://msraig-ubuntu-2:8001"
TEST_DATASET_DIR = "test_dataset"
REPORTS_DIR = "test_dataset/reports"
MAX_ERROR_THRESHOLD = 10.0  # Maximum error in pixels for color mapping


def load_keypoints_from_json(json_path):
    """
    Load keypoints from a JSON file.
    
    Args:
        json_path (str): Path to JSON file containing keypoints
        
    Returns:
        list: List of keypoint dictionaries with 'x' and 'y' coordinates
              Returns None if loading fails
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if 'keypoints' in data:
            keypoints = data['keypoints']
        elif isinstance(data, list):
            keypoints = data
        else:
            print(f"⚠️ Unexpected JSON format in {json_path}")
            return None
        
        # Ensure keypoints have required fields
        valid_keypoints = []
        for kp in keypoints:
            if 'x' in kp and 'y' in kp:
                valid_keypoints.append(kp)
            else:
                print(f"⚠️ Invalid keypoint format in {json_path}")
        
        return valid_keypoints if valid_keypoints else None
        
    except Exception as e:
        print(f"❌ Error loading keypoints from {json_path}: {e}")
        return None


def find_image_json_pairs(directory):
    """
    Find all images with corresponding JSON files in a directory.
    
    Args:
        directory (Path): Directory to search
        
    Returns:
        dict: Dictionary mapping image paths to their JSON keypoint files
              {image_path: json_path}
    """
    pairs = {}
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Find all images in directory
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in image_extensions:
            # Check if corresponding JSON exists
            json_path = file_path.with_suffix('.json')
            if json_path.exists():
                pairs[file_path] = json_path
    
    return pairs


def find_all_images(directory):
    """
    Find all image files in a directory.
    
    Args:
        directory (Path): Directory to search
        
    Returns:
        list: List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in image_extensions:
            images.append(file_path)
    
    return images


def smooth_color_transition(value):
    """
    Create smooth color transition for values from 0 to 1.
    
    Color progression:
    0.0 -> (0,255,0)   Green (good)
    0.25 -> (0,255,255) Cyan  
    0.5 -> (0,0,255)   Blue
    0.75 -> (255,0,255) Magenta
    1.0 -> (255,0,0)   Red (bad)
    
    Args:
        value (float): Value between 0 and 1
        
    Returns:
        tuple: (B, G, R) color values for OpenCV
    """
    # Clamp value to [0, 1]
    value = max(0.0, min(1.0, value))
    
    # Define the 5 key colors in BGR format for OpenCV
    colors_bgr = [
        (0, 255, 0),    # Green (0.0) - good
        (255, 255, 0),  # Cyan (0.25)
        (255, 0, 0),    # Blue (0.5)
        (255, 0, 255),  # Magenta (0.75)
        (0, 0, 255)     # Red (1.0) - bad
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
    color1_bgr = colors_bgr[segment_index]
    color2_bgr = colors_bgr[segment_index + 1]
    
    # Linear interpolation between the two colors
    b = int(color1_bgr[0] * (1 - local_t) + color2_bgr[0] * local_t)
    g = int(color1_bgr[1] * (1 - local_t) + color2_bgr[1] * local_t)
    r = int(color1_bgr[2] * (1 - local_t) + color2_bgr[2] * local_t)
    
    # Return in BGR format for OpenCV
    return (b, g, r)


def consistency_to_color(consistency_distance, max_distance=MAX_ERROR_THRESHOLD):
    """
    Map consistency distance (error) to color using smooth transition.
    
    Args:
        consistency_distance (float): Consistency distance in pixels
        max_distance (float): Maximum distance to map to red (default: from config)
        
    Returns:
        tuple: (B, G, R) color values for OpenCV
    """
    # Normalize consistency distance to [0, 1] range
    normalized_value = min(consistency_distance / max_distance, 1.0)
    return smooth_color_transition(normalized_value)


def draw_keypoints_on_image(image, keypoints, color=(0, 0, 255), marker_size=10, thickness=2, consistency_distances=None, max_error=MAX_ERROR_THRESHOLD):
    """
    Draw keypoints as crosses on an image.
    
    Args:
        image: Image array (will be copied, not modified in place)
        keypoints: List of keypoint dicts with 'x' and 'y' fields
        color: BGR color tuple (default: red (0, 0, 255)) - used if consistency_distances is None
        marker_size: Size of the cross marker in pixels
        thickness: Line thickness
        consistency_distances: Optional list of consistency distances for color mapping
        max_error: Maximum error threshold for color mapping
        
    Returns:
        Image with keypoints drawn
    """
    img_with_kps = image.copy()
    
    for i, kp in enumerate(keypoints):
        x = int(round(kp['x']))
        y = int(round(kp['y']))
        
        # Determine color for this keypoint
        if consistency_distances is not None and i < len(consistency_distances):
            kp_color = consistency_to_color(consistency_distances[i], max_error)
        else:
            kp_color = color
        
        # Draw a cross
        # Horizontal line
        cv2.line(img_with_kps, (x - marker_size, y), (x + marker_size, y), kp_color, thickness)
        # Vertical line
        cv2.line(img_with_kps, (x, y - marker_size), (x, y + marker_size), kp_color, thickness)
    
    return img_with_kps


def generate_visualization_images(subdir_path, subdir_results, output_dir):
    """
    Generate visualization images for tracking results.
    
    For each reference-target pair:
        - Create side-by-side visualization
        - Left: reference image with red crosses
        - Right: target image with green crosses
    
    Args:
        subdir_path (Path): Path to subdirectory containing original images
        subdir_results (dict): Tracking results for this subdirectory
        output_dir (Path): Output directory for generated images
        
    Returns:
        list: List of generated image info dicts
    """
    print("\n🎨 Generating visualization images...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_images = []
    
    # Group results by reference image
    ref_groups = {}
    for result in subdir_results['tracking_results']:
        if not result.get('success'):
            continue
            
        ref_img_name = result['reference_image']
        if ref_img_name not in ref_groups:
            ref_groups[ref_img_name] = []
        ref_groups[ref_img_name].append(result)
    
    # Process each reference group
    for ref_img_name, target_results in ref_groups.items():
        # Process each target for this reference
        for result in target_results:
            # Get paths and keypoints from result
            ref_img_path = Path(result['reference_image_path'])
            target_img_path = Path(result['target_image_path'])
            ref_keypoints = result['reference_keypoints']
            tracked_keypoints = result['tracked_keypoints']
            
            # Load images
            ref_img = cv2.imread(str(ref_img_path))
            target_img = cv2.imread(str(target_img_path))
            
            if ref_img is None or target_img is None:
                print(f"  ⚠️ Could not load images for {ref_img_name} -> {result['target_image']}")
                continue
            
            # Draw red crosses on reference image
            ref_img_with_kps = draw_keypoints_on_image(ref_img, ref_keypoints, color=(0, 0, 255))
            
            # Draw color-coded crosses on target image based on consistency distance
            # Extract consistency distances if available
            consistency_distances = None
            if tracked_keypoints:
                consistency_distances = [kp.get('consistency_distance', 0) for kp in tracked_keypoints]
            
            target_img_with_kps = draw_keypoints_on_image(
                target_img, 
                tracked_keypoints, 
                color=(0, 255, 0),  # Default green if no consistency data
                consistency_distances=consistency_distances,
                max_error=MAX_ERROR_THRESHOLD
            )
            
            # Create side-by-side image
            h1, w1 = ref_img_with_kps.shape[:2]
            h2, w2 = target_img_with_kps.shape[:2]
            
            # Make both images the same height
            max_height = max(h1, h2)
            
            # Resize if needed
            if h1 != max_height:
                scale = max_height / h1
                ref_img_with_kps = cv2.resize(ref_img_with_kps, (int(w1 * scale), max_height))
                w1 = int(w1 * scale)
            
            if h2 != max_height:
                scale = max_height / h2
                target_img_with_kps = cv2.resize(target_img_with_kps, (int(w2 * scale), max_height))
                w2 = int(w2 * scale)
            
            # Create combined image with gap
            gap = 20
            combined_width = w1 + gap + w2
            combined_img = np.zeros((max_height, combined_width, 3), dtype=np.uint8)
            combined_img.fill(255)  # White background for gap
            
            # Place images
            combined_img[:, :w1] = ref_img_with_kps
            combined_img[:, w1+gap:] = target_img_with_kps
            
            # Add labels
            cv2.putText(combined_img, f"Reference: {ref_img_path.name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_img, f"Reference: {ref_img_path.name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            cv2.putText(combined_img, f"Target: {target_img_path.name}", (w1+gap+10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_img, f"Target: {target_img_path.name}", (w1+gap+10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Save combined image
            output_filename = f"{ref_img_path.stem}_to_{target_img_path.stem}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), combined_img)
            
            # Prepare image info
            img_info = {
                'filename': output_filename,
                'reference_image': ref_img_path.name,
                'target_image': target_img_path.name,
                'num_keypoints': result.get('num_keypoints_tracked', 0),
                'success': result.get('success', False)
            }
            
            # Add consistency stats if available
            if 'consistency_stats' in result:
                img_info['consistency_stats'] = result['consistency_stats']
            
            generated_images.append(img_info)
            
            print(f"  ✅ Generated: {output_filename}")
    
    return generated_images


def generate_html_report(subdir_name, subdir_results, generated_images, output_dir, max_error_threshold=MAX_ERROR_THRESHOLD):
    """
    Generate HTML report for a subdirectory.
    
    Args:
        subdir_name (str): Name of subdirectory
        subdir_results (dict): Tracking results
        generated_images (list): List of generated image info
        output_dir (Path): Output directory for HTML file
        max_error_threshold (float): Maximum error threshold for display
        
    Returns:
        Path: Path to generated HTML file
    """
    print("\n📄 Generating HTML report...")
    
    # Calculate statistics
    total_tests = len(subdir_results['tracking_results'])
    successful_tests = sum(1 for r in subdir_results['tracking_results'] if r.get('success'))
    failed_tests = total_tests - successful_tests
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keypoint Tracking Report - {subdir_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-item {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .summary-item.failed {{
            border-left-color: #f44336;
        }}
        .summary-item h3 {{
            margin: 0 0 5px 0;
            color: #666;
            font-size: 14px;
        }}
        .summary-item .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .legend {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #ffc107;
        }}
        .legend h3 {{
            margin-top: 0;
        }}
        .color-box {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border: 1px solid #ccc;
            vertical-align: middle;
        }}
        .red {{ background-color: red; }}
        .green {{ background-color: lime; }}
        .image-pair {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .image-pair h3 {{
            margin-top: 0;
            color: #333;
        }}
        .image-pair img {{
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
        .image-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
            font-size: 14px;
            color: #666;
        }}
        .image-info span {{
            background-color: white;
            padding: 8px;
            border-radius: 3px;
        }}
        .timestamp {{
            color: #999;
            font-size: 14px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Keypoint Tracking Report: {subdir_name}</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <h3>Total Tests</h3>
                    <div class="value">{total_tests}</div>
                </div>
                <div class="summary-item">
                    <h3>Successful</h3>
                    <div class="value">{successful_tests}</div>
                </div>
                <div class="summary-item failed">
                    <h3>Failed</h3>
                    <div class="value">{failed_tests}</div>
                </div>
                <div class="summary-item">
                    <h3>Success Rate</h3>
                    <div class="value">{success_rate:.1f}%</div>
                </div>
                <div class="summary-item">
                    <h3>Reference Images</h3>
                    <div class="value">{subdir_results['num_references']}</div>
                </div>
                <div class="summary-item">
                    <h3>Total Images</h3>
                    <div class="value">{subdir_results['num_total_images']}</div>
                </div>
            </div>
        </div>
        
        <div class="legend">
            <h3>Legend</h3>
            <p>
                <span class="color-box red"></span> <strong>Red crosses:</strong> Reference keypoints (original positions)
            </p>
            <p>
                <strong>Tracked keypoints (color-coded by error):</strong>
            </p>
            <ul style="margin-top: 5px;">
                <li><span class="color-box green"></span> <strong>Green:</strong> Excellent tracking (0.0 - {max_error_threshold*0.25:.1f} pixels error)</li>
                <li><span class="color-box" style="background-color: cyan;"></span> <strong>Cyan:</strong> Good tracking ({max_error_threshold*0.25:.1f} - {max_error_threshold*0.5:.1f} pixels error)</li>
                <li><span class="color-box" style="background-color: blue;"></span> <strong>Blue:</strong> Moderate error ({max_error_threshold*0.5:.1f} - {max_error_threshold*0.75:.1f} pixels error)</li>
                <li><span class="color-box" style="background-color: magenta;"></span> <strong>Magenta:</strong> High error ({max_error_threshold*0.75:.1f} - {max_error_threshold:.1f} pixels error)</li>
                <li><span class="color-box red"></span> <strong>Red:</strong> Very high error (≥ {max_error_threshold:.1f} pixels)</li>
            </ul>
            <p style="margin-top: 10px; font-size: 12px; color: #666;">
                <strong>Note:</strong> Maximum error threshold is set to {max_error_threshold} pixels. 
                Errors beyond this threshold are displayed in red.
            </p>
        </div>
        
        <h2>Tracking Results</h2>
"""
    
    # Add each image pair
    for img_info in generated_images:
        ref_name = img_info['reference_image']
        target_name = img_info['target_image']
        num_kps = img_info['num_keypoints']
        
        # Get consistency stats if available
        consistency_info = ""
        if 'consistency_stats' in img_info:
            stats = img_info['consistency_stats']
            mean_error = stats.get('mean_consistency_distance', 0)
            max_error = stats.get('max_consistency_distance', 0)
            high_acc = stats.get('high_accuracy_count', 0)
            consistency_info = f"""
                <span><strong>Mean error:</strong> {mean_error:.2f} px</span>
                <span><strong>Max error:</strong> {max_error:.2f} px</span>
                <span><strong>High accuracy (&lt;1px):</strong> {high_acc}/{num_kps}</span>
            """
        
        html_content += f"""
        <div class="image-pair">
            <h3>{ref_name} → {target_name}</h3>
            <img src="{img_info['filename']}" alt="Tracking result">
            <div class="image-info">
                <span><strong>Keypoints tracked:</strong> {num_kps}</span>
                <span><strong>Status:</strong> {'✅ Success' if img_info['success'] else '❌ Failed'}</span>
                {consistency_info}
            </div>
        </div>
"""
    
    # Close HTML
    html_content += f"""
        <div class="timestamp">
            <p>Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML file
    html_path = output_dir / "report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"  ✅ HTML report saved: {html_path}")
    
    return html_path


def test_subdirectory(subdir_path, tracker):
    """
    Test keypoint tracking for all images in a subdirectory.
    
    For each reference image with JSON:
        - Track keypoints on all other images (with and without JSON)
    
    Args:
        subdir_path (Path): Path to subdirectory
        tracker: FFPPWebAPIKeypointTracker instance
        
    Returns:
        dict: Results dictionary containing tracking statistics and results
    """
    print(f"\n{'='*80}")
    print(f"📁 Processing subdirectory: {subdir_path.name}")
    print(f"{'='*80}")
    
    # Find images with JSON keypoints (reference images)
    ref_pairs = find_image_json_pairs(subdir_path)
    
    if not ref_pairs:
        print(f"⚠️ No image-JSON pairs found in {subdir_path.name}")
        return None
    
    # Find all images (including those without JSON)
    all_images = find_all_images(subdir_path)
    
    print(f"✅ Found {len(ref_pairs)} reference image(s) with keypoints")
    print(f"✅ Found {len(all_images)} total image(s) in directory")
    
    # Results storage
    results = {
        'subdirectory': subdir_path.name,
        'num_references': len(ref_pairs),
        'num_total_images': len(all_images),
        'tracking_results': []
    }
    
    # Process each reference image
    for ref_idx, (ref_image_path, ref_json_path) in enumerate(ref_pairs.items(), 1):
        print(f"\n{'─'*80}")
        print(f"🎯 Reference {ref_idx}/{len(ref_pairs)}: {ref_image_path.name}")
        print(f"{'─'*80}")
        
        # Load reference image
        ref_img = cv2.imread(str(ref_image_path))
        if ref_img is None:
            print(f"❌ Failed to load reference image: {ref_image_path}")
            continue
        
        # Load reference keypoints
        ref_keypoints = load_keypoints_from_json(ref_json_path)
        if ref_keypoints is None:
            print(f"❌ Failed to load keypoints from: {ref_json_path}")
            continue
        
        print(f"✅ Loaded reference: {ref_img.shape[1]}x{ref_img.shape[0]}")
        print(f"✅ Loaded {len(ref_keypoints)} keypoints")
        
        # Set reference in tracker
        ref_name = f"{subdir_path.name}_{ref_image_path.stem}"
        ref_start_time = time.time()
        ref_result = tracker.set_reference_image(ref_img, ref_keypoints, ref_name)
        ref_time = time.time() - ref_start_time
        
        if not ref_result.get('success'):
            print(f"❌ Failed to set reference: {ref_result.get('error')}")
            continue
        
        print(f"✅ Reference set in {ref_time:.3f}s")
        
        # Track on all other images
        target_images = [img for img in all_images if img != ref_image_path]
        
        print(f"\n📊 Tracking on {len(target_images)} target image(s)...")
        
        for target_idx, target_image_path in enumerate(target_images, 1):
            # Load target image
            target_img = cv2.imread(str(target_image_path))
            if target_img is None:
                print(f"❌ Failed to load target image: {target_image_path}")
                continue
            
            # Check if target has JSON (for comparison)
            target_json_path = target_image_path.with_suffix('.json')
            has_target_json = target_json_path.exists()
            
            # Track keypoints
            track_start_time = time.time()
            track_result = tracker.track_keypoints(
                target_img, 
                reference_name=ref_name,
                bidirectional=True
            )
            track_time = time.time() - track_start_time
            
            if track_result.get('success'):
                tracked_kps = len(track_result.get('tracked_keypoints', []))
                service_time = track_result.get('total_processing_time', 0)
                
                # Calculate statistics from bidirectional validation
                consistency_info = ""
                if 'bidirectional_stats' in track_result:
                    stats = track_result['bidirectional_stats']
                    mean_consistency = stats.get('mean_consistency_distance', 0)
                    high_acc = stats.get('high_accuracy_count', 0)
                    consistency_info = f" | Consistency: {mean_consistency:.2f}px ({high_acc} high-acc)"
                
                print(f"  [{target_idx}/{len(target_images)}] ✅ {target_image_path.name:<40} | "
                      f"Tracked: {tracked_kps:2d} | Time: {track_time:.3f}s | "
                      f"Has JSON: {'Yes' if has_target_json else 'No '}{consistency_info}")
                
                # Store result
                result_entry = {
                    'reference_image': ref_image_path.name,
                    'reference_image_path': str(ref_image_path),
                    'reference_keypoints': ref_keypoints,  # Store reference keypoints
                    'target_image': target_image_path.name,
                    'target_image_path': str(target_image_path),
                    'target_has_json': has_target_json,
                    'tracked_keypoints': track_result.get('tracked_keypoints', []),  # Store tracked keypoints
                    'success': True,
                    'num_keypoints_tracked': tracked_kps,
                    'api_call_time': track_time,
                    'service_processing_time': service_time,
                    'bidirectional_enabled': True
                }
                
                if 'bidirectional_stats' in track_result:
                    result_entry['consistency_stats'] = track_result['bidirectional_stats']
                
                results['tracking_results'].append(result_entry)
                
            else:
                print(f"  [{target_idx}/{len(target_images)}] ❌ {target_image_path.name:<40} | "
                      f"Error: {track_result.get('error', 'Unknown')}")
                
                results['tracking_results'].append({
                    'reference_image': ref_image_path.name,
                    'target_image': target_image_path.name,
                    'target_has_json': has_target_json,
                    'success': False,
                    'error': track_result.get('error', 'Unknown')
                })
        
        # Clean up reference
        cleanup_result = tracker.remove_reference_image(ref_name)
        if cleanup_result.get('success'):
            print("✅ Reference cleaned up")
    
    return results


def run_test_dataset(service_url=WEB_SERVICE_URL, max_error_threshold=MAX_ERROR_THRESHOLD, skip_existing=False):
    """
    Run tests on all subdirectories in test_dataset.
    
    Args:
        service_url (str): URL of FlowFormer++ web service
        max_error_threshold (float): Maximum error threshold in pixels for color mapping
        skip_existing (bool): If True, skip subdirectories that already have reports
        
    Returns:
        dict: Complete test results
    """
    print("🧪 Test Dataset Runner")
    print("=" * 80)
    print(f"Service URL: {service_url}")
    print("Bidirectional validation: Enabled")
    print(f"Max error threshold: {max_error_threshold} pixels")
    print(f"Skip existing reports: {'Yes' if skip_existing else 'No'}")
    print(f"Dataset directory: {TEST_DATASET_DIR}")
    print("=" * 80)
    
    # Initialize tracker
    print("\n🚀 Initializing FFPPWebAPIKeypointTracker...")
    try:
        tracker = FFPPWebAPIKeypointTracker(service_url=service_url)
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return None
    
    # Check service health
    print("🏥 Checking service health...")
    health = tracker.get_service_health()
    if not health.get('success'):
        print(f"❌ Service not available: {health.get('error')}")
        return None
    
    print(f"✅ Service available: {health.get('status')}")
    
    # Find all subdirectories in test_dataset
    dataset_path = Path(TEST_DATASET_DIR)
    if not dataset_path.exists():
        print(f"❌ Test dataset directory not found: {TEST_DATASET_DIR}")
        return None
    
    # Get all subdirectories, excluding the reports directory
    reports_dir_name = Path(REPORTS_DIR).name
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name != reports_dir_name]
    
    if not subdirs:
        print(f"⚠️ No subdirectories found in {TEST_DATASET_DIR}")
        return None
    
    print(f"\n✅ Found {len(subdirs)} subdirectory(ies) to process")
    
    # Process each subdirectory
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'service_url': service_url,
        'bidirectional_enabled': True,
        'subdirectories': []
    }
    
    start_time = time.time()
    
    for idx, subdir in enumerate(sorted(subdirs), 1):
        # Check if report already exists and skip_existing is enabled
        report_subdir = Path(REPORTS_DIR) / subdir.name
        if skip_existing and report_subdir.exists():
            report_html = report_subdir / "report.html"
            if report_html.exists():
                print(f"\n⏭️  [{idx}/{len(subdirs)}] Skipping '{subdir.name}' (report already exists)")
                continue
        
        print(f"\n🔍 [{idx}/{len(subdirs)}] Processing: {subdir.name}")
        
        subdir_results = test_subdirectory(subdir, tracker)
        if subdir_results:
            all_results['subdirectories'].append(subdir_results)
            
            # Generate report for this subdirectory
            report_output_dir = Path(REPORTS_DIR) / subdir.name
            
            # Generate visualization images
            generated_images = generate_visualization_images(subdir, subdir_results, report_output_dir)
            
            # Generate HTML report
            html_path = generate_html_report(subdir.name, subdir_results, generated_images, report_output_dir, max_error_threshold)
            
            # Store report path in results
            subdir_results['report_path'] = str(html_path)
            subdir_results['num_generated_images'] = len(generated_images)
    
    total_time = time.time() - start_time
    all_results['total_processing_time'] = total_time
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 Test Summary")
    print(f"{'='*80}")
    
    total_tracking_tests = 0
    successful_tests = 0
    
    for subdir_result in all_results['subdirectories']:
        subdir_name = subdir_result['subdirectory']
        num_results = len(subdir_result['tracking_results'])
        num_success = sum(1 for r in subdir_result['tracking_results'] if r.get('success'))
        report_path = subdir_result.get('report_path', 'N/A')
        
        total_tracking_tests += num_results
        successful_tests += num_success
        
        print(f"  {subdir_name}: {num_success}/{num_results} successful")
        print(f"    Report: {report_path}")
    
    print(f"\n  Total tracking tests: {total_tracking_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Failed: {total_tracking_tests - successful_tests}")
    print(f"  Success rate: {(successful_tests/total_tracking_tests*100):.1f}%")
    print(f"  Total time: {total_time:.2f}s")
    
    all_results['summary'] = {
        'total_tracking_tests': total_tracking_tests,
        'successful_tests': successful_tests,
        'failed_tests': total_tracking_tests - successful_tests,
        'success_rate': successful_tests / total_tracking_tests if total_tracking_tests > 0 else 0
    }
    
    return all_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run keypoint tracking tests on test_dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_test_dataset.py
  python tests/run_test_dataset.py --url http://server:8001
  python tests/run_test_dataset.py --max-error 15.0
  python tests/run_test_dataset.py --skip-existing
  python tests/run_test_dataset.py -s --max-error 8.0
        """
    )
    
    parser.add_argument('--url', '-u', type=str, 
                        help='FlowFormer++ web service URL (default: http://localhost:8001)')
    parser.add_argument('--max-error', '-m', type=float, default=MAX_ERROR_THRESHOLD,
                        help=f'Maximum error threshold in pixels for color mapping (default: {MAX_ERROR_THRESHOLD})')
    parser.add_argument('--skip-existing', '-s', action='store_true',
                        help='Skip subdirectories that already have reports generated')
    
    args = parser.parse_args()
    
    # Use provided URL or default
    service_url = args.url if args.url else WEB_SERVICE_URL
    
    # Run tests
    results = run_test_dataset(
        service_url=service_url, 
        max_error_threshold=args.max_error,
        skip_existing=args.skip_existing
    )
    
    if results is None:
        print("\n❌ Test run failed - check errors above")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("✅ Test dataset run completed successfully!")
    print(f"{'='*80}")
    print(f"\n📊 Reports generated in: {REPORTS_DIR}/")
    print("   Open the report.html files in your browser to view results")
    
    # List all generated reports
    reports_path = Path(REPORTS_DIR)
    if reports_path.exists():
        for subdir_result in results.get('subdirectories', []):
            if 'report_path' in subdir_result:
                print(f"   - {subdir_result['report_path']}")


if __name__ == "__main__":
    main()
