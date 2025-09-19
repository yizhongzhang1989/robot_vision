````markdown
# Robot Vision

High-performance keypoint tracking for robotics applications using FlowFormer++.

## Quick Start

```bash
# Install
pip install -e .

# Run the comprehensive example and test suite
conda run --name flowformerpp python examples/ffpp_keypoint_tracker_example.py

# Use in Python  
from core.ffpp_keypoint_tracker import FFPPKeypointTracker
tracker = FFPPKeypointTracker()
tracker.set_reference_image(ref_image, keypoints)
result = tracker.track_keypoints(target_image)
```

## Examples

- `examples/ffpp_keypoint_tracker_example.py` - Comprehensive example demonstrating:
  - Basic keypoint tracking with simplified interface
  - Bidirectional flow validation for accuracy assessment
  - Multiple reference image management
  - Performance benchmarking

## Structure

```
robot_vision/
├── core/                          # ✨ Main core functionality
│   ├── __init__.py                # Core module exports
│   ├── ffpp_keypoint_tracker.py   # High-performance FlowFormer++ tracker
│   ├── keypoint_tracker.py        # Original keypoint tracker
│   └── utils.py                   # Utility functions
├── examples/                      # ✨ Usage examples and demos
│   ├── ffpp_keypoint_tracker_example.py  # Comprehensive test suite
│   ├── keypoint_tracker_origin.py        # Original examples
│   └── keypoint_tracker_simple.py        # Simplified examples
├── sample_data/                   # Sample images and keypoints
│   └── flow_image_pair/          # Test data for tracking
├── output/                        # Generated results and visualizations
├── ThirdParty/                    # External dependencies
│   └── FlowFormerPlusPlusServer/ # FlowFormer++ server
├── README.md                      # This file
├── setup.py                       # Package setup
└── requirements.txt               # Dependencies
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- FlowFormer++ dependencies (torch, torchvision, etc.)
- OpenCV for image processing

## Features

### FFPPKeypointTracker
- **21x faster** than API-based tracking (~0.3s vs ~7s)
- **Simplified interface**: Just `set_reference_image()` + `track_keypoints()`
- **Bidirectional flow validation** for accuracy assessment
- **Multiple reference management** with automatic coordinate scaling
- **GPU acceleration** with CUDA support

### Usage Patterns

```python
# Basic tracking
tracker = FFPPKeypointTracker()
tracker.set_reference_image(ref_image, keypoints)
result = tracker.track_keypoints(target_image)

# With bidirectional validation
result = tracker.track_keypoints(target_image, bidirectional=True)
consistency = result['bidirectional_stats']['mean_consistency_distance']

# Multiple references
tracker.set_reference_image(img1, kpts1, image_name="setup1")
tracker.set_reference_image(img2, kpts2, image_name="setup2")
result = tracker.track_keypoints(target, reference_name="setup1")
```

A repository for vision algorithm development for robotics applications.

This repository contains computer vision algorithms and tools specifically designed for robotic systems. The project focuses on developing robust vision solutions for various robotic tasks including object detection, tracking, localization, and navigation.
