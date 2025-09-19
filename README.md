# Robot Vision

High-performance keypoint tracking for robotics applications using FlowFormer++.

## Quick Start

```bash
# One-click setup (recommended)
./setup_all_in_one.sh

# Or manual installation
pip install -r requirements.txt
pip install -e .

# Run examples
python examples/ffpp_keypoint_tracker_example.py
```

```python
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
- NumPy ≥1.21.0 (supports both 1.x and 2.x)
- All dependencies auto-resolved with `setup_all_in_one.sh`

## Features

### FFPPKeypointTracker
- **21x faster** than API-based tracking (~0.3s vs ~7s)
- **NumPy 2.x compatible** - automatic compatibility fixes
- **One-click setup** - handles submodules, models, dependencies
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

## Installation

### Option 1: One-click Setup (Recommended)
```bash
git clone --recursive https://github.com/yizhongzhang1989/robot_vision.git
cd robot_vision
./setup_all_in_one.sh
```
Automatically handles:
- Git submodules
- NumPy compatibility fixes  
- Model downloads
- All dependencies

### Option 2: Manual Setup
```bash
pip install -r requirements.txt  # NumPy 2.x compatible versions
pip install -e .
# Download models: cd ThirdParty/FlowFormerPlusPlusServer && ./scripts/download_ckpts.sh
```
