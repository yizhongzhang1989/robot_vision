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

## API Server Usage

### Start Server
```bash
python app.py  # Runs on http://0.0.0.0:8009
```

### API Endpoints

**Health Check**
```bash
curl -X GET http://localhost:8009/health
```

**Set Reference Image**
```bash
curl -X POST http://localhost:8009/set_reference \
  -F "image=@path/to/image.jpg" \
  -F "keypoints=[{\"x\": 100, \"y\": 150}, {\"x\": 200, \"y\": 250}]" \
  -F "image_name=ref1"
```

**Track Keypoints**
```bash
curl -X POST http://localhost:8009/track_keypoints \
  -F "image=@path/to/target.jpg" \
  -F "bidirectional=true"
```

**List References**
```bash
curl -X GET http://localhost:8009/references
```

### Python Client
```python
import requests
base_url = 'http://localhost:8009'

# Health check
requests.get(f'{base_url}/health').json()

# Set reference
files = {'image': open('ref.jpg', 'rb')}
data = {'keypoints': '[{"x": 100, "y": 150}]', 'image_name': 'ref1'}
requests.post(f'{base_url}/set_reference', files=files, data=data).json()

# Track keypoints
files = {'image': open('target.jpg', 'rb')}
data = {'bidirectional': 'true'}
requests.post(f'{base_url}/track_keypoints', files=files, data=data).json()
```

### Interactive Documentation
Visit `http://localhost:8009/docs` for full API documentation and testing interface.
