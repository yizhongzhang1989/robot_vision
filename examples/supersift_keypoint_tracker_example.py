#!/usr/bin/env python3
"""
SuperSiftKeypointTracker Example and Test Suite

This example demonstrates the usage of SuperSiftKeypointTracker

Code Structure:
- load_sample_data(): Returns (target_img, ref_img, ref_keypoints) tuple
- test_basic_tracking(): Demonstrates core two-step tracking process

Output Location:
- All results saved to: output/superpoint_keypoint_tracker_example_output/
- Includes visualizations

Usage:
    python examples/supersift_keypoint_tracker_example.py
"""

import os
import cv2
import json
import numpy as np
import time
import torch

# Add the parent directory to the path to import core modules
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.supersift_keypoint_tracker import SuperSiftKeypointTracker


def load_sample_data(references_path="sample_data/supersift_sample", test_images_path="sample_data/supersift_sample"):
    """
    Load sample data for testing.
    Args:
        references_path: Path to the reference image folder
        test_images_path: Path to the target image folder
    Returns:
        references: (Reference image, keypoint) (list of tuples)
        test_images: List of target images
    """
    template_images_info = []

    json_files = [f for f in os.listdir(references_path) if f.endswith(".json")]
    for json_file in json_files:
        json_path = os.path.join(references_path, json_file)
        image_path = json_path.replace(".json", ".jpg")
        if not os.path.exists(image_path):
            raise ValueError(f"Image file {image_path} not found for JSON file {json_file}")
        with open(json_path, "r") as f:
            data = json.load(f)
        keypoints = data["keypoints"]
        template_images_info.append((cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), keypoints, os.path.basename(image_path)))

    test_image_files = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if f.endswith(".jpg") and not os.path.exists(os.path.join(test_images_path, f.replace(".jpg", ".json")))]
    test_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in test_image_files]

    return template_images_info, test_images


def test_basic_tracking():
    # ========================================
    # DATA PREPARATION
    # ========================================
    template_images_info, test_images = load_sample_data()

    # ========================================
    # TRACKER INITIALIZATION
    # ========================================
    print("\n🚀 Initializing FFPPKeypointTracker...")
    init_start_time = time.time()
    tracker = SuperSiftKeypointTracker()
    init_elapsed_time = time.time() - init_start_time

    if not tracker.model_loaded:
        print("❌ Failed to load model")
        return False

    print(f"✅ FFPPKeypointTracker initialized on {tracker.device}")
    print(f"   Initialization time: {init_elapsed_time:.3f}s")

    for ref_img, ref_keypoints, ref_name in template_images_info:
        tracker.set_reference_image(ref_img, ref_keypoints, image_name=ref_name)

    # Create output directory
    output_dir = "output/suerpoint_keypoint_tracker_example_output"
    os.makedirs(output_dir, exist_ok=True)

    print("\n🎯 Step 2: Tracking keypoints in target image...")

    for id, target_img in enumerate(test_images):
        result = tracker.track_keypoints(target_img)
        if not result["success"]:
            print(f"❌ Keypoint tracking failed: {result.get('error', 'Unknown error')}")
            continue
        else:
            print(f"✅ Keypoint tracking succeeded: {len(result['tracked_keypoints'])} keypoints tracked.")
            print(f"   Tracking time: {result['processing_time']:.3f}s")

        # ========================================
        # OUTPUT AND VISUALIZATION
        # ========================================
        try:
            # 1. SAVE JSON RESULTS FIRST
            json_path = os.path.join(output_dir, f"basic_tracking_results_{id}.json")

            # Write the result directly as returned by track_keypoints()
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"✅ Results saved: {json_path}")

            # 2. CREATE AND SAVE VISUALIZATION IMAGE
            vis_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
            h, w = vis_img.shape[:2]

            for i, kp in enumerate(result["tracked_keypoints"]):
                x, y = int(round(kp["x"])), int(round(kp["y"]))
                dev = kp["deviation"]
                name = kp["name"]

                # use color coding to map dev from green to yellow, if dis < 5.0
                if dev < 1.0:
                    color = (0, 255, 0)
                elif dev < 3.0:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                if 0 <= x < w and 0 <= y < h:
                    cv2.drawMarker(vis_img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    cv2.putText(vis_img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # plot color legend
            # Draw legend rectangle at top right corner
            legend_width, legend_height = 240, 70
            legend_x1, legend_y1 = w - legend_width - 10, 10
            legend_x2, legend_y2 = w - 10, 10 + legend_height
            cv2.rectangle(vis_img, (legend_x1, legend_y1), (legend_x2, legend_y2), (50, 50, 5), -1)
            # Put legend text inside the rectangle
            cv2.putText(vis_img, "   Dev < 1.0", (legend_x1 + 20, legend_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis_img, "   1.0 <= Dev < 3.0", (legend_x1 + 20, legend_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis_img, "   Dev >= 3.0", (legend_x1 + 20, legend_y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # put cross marker left to each text
            cv2.drawMarker(vis_img, (legend_x1 + 10, legend_y1 + 15), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            cv2.drawMarker(vis_img, (legend_x1 + 10, legend_y1 + 35), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            cv2.drawMarker(vis_img, (legend_x1 + 10, legend_y1 + 55), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

            vis_path = os.path.join(output_dir, f"basic_tracking_visualization_{id}.jpg")
            cv2.imwrite(vis_path, vis_img)

            print(f"✅ Visualization saved: {vis_path}")

        except Exception as e:
            print(f"⚠️ Output creation failed: {e}")


def main():
    print("========================================")
    print(" SuperSiftKeypointTracker Example/Test ")
    print("========================================")

    test_basic_tracking()


if __name__ == "__main__":
    main()
