# Robot Vision

A comprehensive robotics vision toolkit providing high-performance keypoint tracking and image annotation capabilities. This repository combines state-of-the-art optical flow techniques with practical tools for robot vision applications.

## 💻 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (required for FlowFormer++)
  - ✅ Tested on: RTX 3090, RTX 4090
  - Minimum: 8GB VRAM recommended
  - CUDA 11.3+ compatible GPU

### Software Requirements
- **OS**: Tested on: Ubuntu 22.04, Ubuntu 24.04
- **Python**: 3.8+
- **CUDA**: 11.3 or later
- **Conda**: Anaconda or Miniconda
- **Git**: With submodule support

## 🚀 Quick Start

### Installation

```bash
# Standard setup
bash setup_all_in_one.sh

# What it does:
# 1. Checks system requirements (stops if Conda is missing)
# 2. Initializes Git submodules
# 3. Creates 'robot_vision' Conda environment with Python 3.8
# 4. Installs all dependencies
# 5. Downloads FlowFormer++ models (~2GB)
# 6. Validates installation by running example

# Activate the environment
conda activate robot_vision

# Start all services
python start_services.py

# Access the dashboard
# Gateway: http://localhost:8000
# FlowFormer++ Tracking: http://localhost:8001
```

### Setup Options

```bash
# View help and usage information
bash setup_all_in_one.sh --help

# Use system Python instead of Conda (not recommended and not tested)
bash setup_all_in_one.sh --skip-conda
```

### Validation

The setup automatically validates the installation by running the FlowFormer++ example.
You can manually re-run validation anytime:

```bash
# Run validation test
bash scripts/run_tests.sh

# This will:
# - Load FlowFormer++ models
# - Process sample images
# - Track keypoints with multiple algorithms
# - Generate visualizations
# - Take about 10-15 seconds
```

## 📖 Usage

### Python API

Direct library integration for best performance (~0.3s per frame with GPU).

```python
from core.ffpp_keypoint_tracker import FFPPKeypointTracker

tracker = FFPPKeypointTracker()
tracker.set_reference_image(ref_image, keypoints)
result = tracker.track_keypoints(target_image)
```

See `examples/ffpp_keypoint_tracker_example.py` for detailed examples.

### Web API

HTTP REST API for remote access and language flexibility (~0.5s per frame).

**Start the web services:**
```bash
conda activate robot_vision
python start_services.py
```

**Python client:**
```python
from core.ffpp_webapi_keypoint_tracker import FFPPWebAPIKeypointTracker

tracker = FFPPWebAPIKeypointTracker(api_url='http://localhost:8001')
tracker.set_reference_image(ref_image, keypoints)
result = tracker.track_keypoints(target_image)
```

See `examples/ffpp_webapi_keypoint_tracker_example.py` for detailed examples.

**API Endpoints:**
- Gateway Dashboard: `http://localhost:8000`
- FlowFormer++ API: `http://localhost:8001`
- API Documentation: `http://localhost:8001/docs`

## 📚 Examples

- **`examples/ffpp_keypoint_tracker_example.py`** - Python API demonstration
  - Basic tracking, bidirectional validation, multiple references, performance benchmarking

- **`examples/ffpp_webapi_keypoint_tracker_example.py`** - Web API demonstration
  - HTTP client usage, real-time dashboard updates, image persistence
