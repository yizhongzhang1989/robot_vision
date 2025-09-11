# Robot Vision

Simple keypoint tracking for robotics applications.

## Quick Start

```bash
# Install
pip install -e .

# Use in Python  
python -c "import robot_vision; robot_vision.track_keypoints()"

# Or use the core module directly
python -c "from core.keypoint_tracker import KeypointTracker; KeypointTracker().run_tracking()"
```

## Structure

```
robot_vision/
├── core/                   # ✨ Main core functionality
│   ├── __init__.py         # Core module exports
│   ├── keypoint_tracker.py # Keypoint tracking implementation
│   └── utils.py           # Utility functions
├── robot_vision/          # Package wrapper (for compatibility)
├── test_data/             # Sample data
├── ThirdParty/            # FlowFormer server
├── README.md              # This file
├── LICENSE                # MIT License
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
├── .gitignore             # Git ignore rules
└── .gitmodules            # Git submodules
```

## Requirements

- Python 3.8+
- FlowFormer server running on http://msraig-ubuntu-3:5000

## Usage

The module tracks keypoints between two images using optical flow:

1. Loads keypoints from `test_data/ref_img_knobs.json`
2. Computes optical flow between `ref_img.jpg` and `comp_img.jpg`  
3. Tracks keypoint movement
4. Saves results to `test_data/output/`

That's it!

A repository for vision algorithm development for robotics applications.

This repository contains computer vision algorithms and tools specifically designed for robotic systems. The project focuses on developing robust vision solutions for various robotic tasks including object detection, tracking, localization, and navigation.
