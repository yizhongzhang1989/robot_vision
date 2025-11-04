# Robot Vision

A comprehensive robotics vision toolkit providing high-performance keypoint tracking and image annotation capabilities. This repository combines state-of-the-art optical flow techniques with practical tools for robot vision applications.

## 📖 Overview

**Robot Vision** is a production-ready computer vision system designed for robotics applications. It provides:

- **🎯 High-Performance Keypoint Tracking**: FlowFormer++ based optical flow tracking with 21x speed improvement over API-based methods (~0.3s vs ~7s per image)
- **🏷️ Interactive Image Labeling**: Web-based tool for annotating images and marking keypoints for training and reference data creation
- **📊 Real-Time Monitoring Dashboard**: Live visualization with Server-Sent Events (SSE) for tracking performance and results
- **🌐 Microservices Architecture**: Independent gateway, tracking, and labeling services with centralized control

### Use Cases
- Robot manipulation and grasping (tracking object keypoints)
- Visual servoing and calibration
- Object pose estimation
- Dataset annotation and preparation
- Real-time vision system monitoring

## 💻 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (required for FlowFormer++)
  - ✅ Tested on: RTX 3090, RTX 4090
  - Minimum: 8GB VRAM recommended
  - CUDA 11.3+ compatible GPU

### Software Requirements
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Python**: 3.8+
- **CUDA**: 11.3 or later
- **Conda**: Anaconda or Miniconda
- **Git**: With submodule support

## 🚀 Quick Start

### Prerequisites

This project **requires Conda** (Anaconda or Miniconda) for proper environment management.

**If you don't have Conda installed:**

```bash
# Option 1: Use our automated installer (Easiest - Recommended)
bash scripts/install_conda.sh

# Option 2: Quick command-line installation
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init
source ~/.bashrc

# Option 3: Interactive installation
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts to customize installation
```

> **Note**: If you really need to use system Python (not recommended), you can run:
> ```bash
> bash setup_all_in_one.sh --skip-conda
> ```
> This may cause dependency conflicts and is only for advanced users.

### Installation

The setup script automatically handles everything:

```bash
# Standard setup (uses Conda - recommended)
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

# Use system Python instead of Conda (not recommended)
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

### Python API
```python
from core.ffpp_keypoint_tracker import FFPPKeypointTracker

tracker = FFPPKeypointTracker()
tracker.set_reference_image(ref_image, keypoints)
result = tracker.track_keypoints(target_image)
```

### Web API Client
```python
from core.ffpp_webapi_keypoint_tracker import FFPPKeypointTracker

# Connect to web service
tracker = FFPPKeypointTracker(api_url='http://localhost:8001')
tracker.set_reference_image(ref_image, keypoints)
result = tracker.track_keypoints(target_image)
```

## 🎯 Features

### Real-Time Web Dashboard
- **Live monitoring** - Server-Sent Events (SSE) for real-time updates
- **16:9 optimized** - Big screen monitoring dashboard
- **Breathing keypoints** - Animated keypoint visualization
- **Responsive scaling** - Keypoints scale with image display
- **Image history** - Stores all processed images and results

### High-Performance Tracking
- **21x faster** than API-based tracking (~0.3s vs ~7s)
- **NumPy 2.x compatible** - automatic compatibility fixes
- **GPU acceleration** - CUDA support for maximum performance
- **Bidirectional validation** - accuracy assessment
- **Multiple references** - manage multiple reference images

## 📁 Project Structure

```
robot_vision/
├── core/                          # ✨ Core tracking functionality
│   ├── ffpp_keypoint_tracker.py   # Local FlowFormer++ tracker
│   ├── ffpp_webapi_keypoint_tracker.py  # Web API client
│   ├── keypoint_tracker.py        # Original tracker
│   └── utils.py                   # Utilities
├── web/                           # 🌐 Web services
│   ├── ffpp_keypoint_tracking/    # Main tracking service
│   │   ├── app.py                 # Flask server (port 8001)
│   │   ├── templates/dashboard.html  # Real-time dashboard
│   │   └── static/                # CSS, JavaScript
│   ├── gateway/                   # Control center (port 8000)
│   └── image_labeling/            # Labeling tool (port 8002)
├── examples/                      # 📚 Usage examples
│   ├── ffpp_keypoint_tracker_example.py  # Local tracker demo
│   ├── ffpp_webapi_keypoint_tracker_example.py  # Web API demo
│   └── keypoint_tracker_*.py     # Other examples
├── config/                        # ⚙️ Configuration
│   └── services.yaml              # Service ports and settings
├── scripts/                       # 🔧 Setup and utilities
│   ├── manage_services.py         # Service management
│   └── setup_*.sh                 # Setup scripts
├── sample_data/                   # Sample images and keypoints
├── output/                        # Generated results
│   └── api_images/                # Dashboard image storage
├── ThirdParty/                    # External dependencies
│   └── FlowFormerPlusPlusServer/  # FlowFormer++ backend
├── setup_all_in_one.sh           # 🎬 One-click setup
├── start_services.py              # 🚀 Start all services
└── requirements.txt               # Dependencies
```

## 🛠️ Installation

### Option 1: One-Click Setup (Recommended)
```bash
git clone --recursive https://github.com/yizhongzhang1989/robot_vision.git
cd robot_vision
bash setup_all_in_one.sh
```

## 📦 Detailed Setup

### Setup Script Breakdown

The `setup_all_in_one.sh` script performs these steps:

**Without --skip-conda flag (Recommended - 6 steps):**
1. **Check system requirements** - Verifies Python, Git, Conda, disk space
   - **Stops immediately** if Conda is not installed
   - Provides detailed installation instructions
2. **Setup Git submodules** - Initializes FlowFormer++, calibration toolkit, labeling tool
3. **Create Conda environment** - Creates `robot_vision` environment with Python 3.8
4. **Install dependencies** - Installs all Python packages in isolated environment
5. **Download models** - Downloads FlowFormer++ checkpoints (~2GB)
6. **Validate installation** - Runs example to ensure everything works

**With --skip-conda flag (Not Recommended - 5 steps):**
- Skips step 3 (Conda environment creation)
- Uses system Python instead
- May cause dependency conflicts

### Individual Setup Scripts

You can run individual setup steps if needed:

```bash
# Check requirements only
bash scripts/check_requirements.sh

# Setup submodules only
bash scripts/setup_submodules.sh update

# Create/manage Conda environment
bash scripts/setup_conda.sh create       # Create environment
bash scripts/setup_conda.sh info         # Show environment info
bash scripts/setup_conda.sh remove       # Remove environment

# Install dependencies only
bash scripts/install_dependencies.sh install

# Download models only
bash scripts/download_models.sh download
bash scripts/download_models.sh status   # Check model status
bash scripts/download_models.sh list     # List available models

# Run validation test
bash scripts/run_tests.sh
```

### Option 2: Manual Setup (Advanced Users)

If you prefer manual control:

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/yizhongzhang1989/robot_vision.git
cd robot_vision

# 2. Create Conda environment
conda create -n robot_vision python=3.8 -y
conda activate robot_vision

# 3. Install dependencies
pip install -r requirements.txt
pip install -r ThirdParty/FlowFormerPlusPlusServer/requirements.txt
pip install -r ThirdParty/camera_calibration_toolkit/requirements.txt
pip install -e .

# 4. Download FlowFormer++ models
cd ThirdParty/FlowFormerPlusPlusServer
./scripts/download_ckpts.sh
cd ../..

# 5. Validate installation
python examples/ffpp_keypoint_tracker_example.py
```

## 🌐 Web Services

### Starting Services

```bash
# Activate the conda environment
conda activate robot_vision

# Start all services at once
python start_services.py

# Services will start on configured ports (from config/services.yaml):
# - Gateway (Control Center): http://localhost:8000
# - FlowFormer++ Tracking: http://localhost:8001
# - Image Labeling Tool: http://localhost:8002
```

### Configuring Ports

Edit `config/services.yaml` to change ports:
```yaml
gateway:
  port: 8000

services:
  ffpp_keypoint_tracking:
    port: 8001
  image_labeling:
    port: 8002
```

Services automatically read configuration on startup - no code changes needed!

### Real-Time Dashboard

Access the monitoring dashboard at `http://localhost:8001` after starting services.

**Features:**
- 📊 Live API call monitoring with SSE (Server-Sent Events)
- 🖼️ Side-by-side reference and target image display
- 📍 Responsive keypoint visualization with breathing animation
- 🎨 16:9 optimized layout for big screens
- 💾 Automatic image storage in `output/api_images/`
- 🔢 Unlimited API call counter (no 50-call limit)

### Web API Usage

#### Python Client

```python
from core.ffpp_webapi_keypoint_tracker import FFPPKeypointTracker

# Initialize with service URL
tracker = FFPPKeypointTracker(api_url='http://localhost:8001')

# Set reference image
tracker.set_reference_image(ref_image, keypoints, image_name='reference_1')

# Track keypoints
result = tracker.track_keypoints(
    target_image,
    bidirectional=True,
    visualize_paths=True
)

# Access results
tracked_keypoints = result['tracked_keypoints']
visualization = result['visualization']  # Returns image
```

#### HTTP Endpoints

**Health Check**
```bash
curl http://localhost:8001/health
```

**Set Reference Image**
```bash
curl -X POST http://localhost:8001/set_reference \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,...",
    "keypoints": [{"x": 100, "y": 150}, {"x": 200, "y": 250}],
    "image_name": "ref1"
  }'
```

**Track Keypoints**
```bash
curl -X POST http://localhost:8001/track_keypoints \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,...",
    "reference_name": "ref1",
    "bidirectional": true,
    "visualize_paths": true
  }'
```

**Dashboard (Real-time monitoring)**
```bash
# Open in browser
http://localhost:8001/
```

**API Events (SSE)**
```bash
# Real-time event stream
curl -N http://localhost:8001/api_events
```

## 💻 Local API Usage

### Basic Tracking

```python
from core.ffpp_keypoint_tracker import FFPPKeypointTracker

# Initialize tracker
tracker = FFPPKeypointTracker(device='cuda')  # or 'cpu'

# Set reference image with keypoints
tracker.set_reference_image(ref_image, keypoints)

# Track keypoints in target image
result = tracker.track_keypoints(target_image)

# Access tracked points
tracked_keypoints = result['tracked_keypoints']
success = result['success']
```

### Advanced Features

```python
# Bidirectional validation for accuracy
result = tracker.track_keypoints(
    target_image,
    bidirectional=True,
    consistency_threshold=5.0
)

consistency = result['bidirectional_stats']['mean_consistency_distance']
reliable_points = result['bidirectional_stats']['reliable_keypoints']

# Multiple reference images
tracker.set_reference_image(img1, kpts1, image_name="setup_1")
tracker.set_reference_image(img2, kpts2, image_name="setup_2")

# Track using specific reference
result = tracker.track_keypoints(target, reference_name="setup_1")

# List available references
references = tracker.list_references()
```

## 📚 Examples

- **`examples/ffpp_keypoint_tracker_example.py`** - Local tracker demonstration
  - Basic tracking
  - Bidirectional validation
  - Multiple references
  - Performance benchmarking

- **`examples/ffpp_webapi_keypoint_tracker_example.py`** - Web API demonstration
  - HTTP API client usage
  - Real-time dashboard updates
  - Image persistence

## 🔧 Configuration

### Services Configuration (`config/services.yaml`)

```yaml
gateway:
  port: 8000
  title: "Robot Vision Control Center"

services:
  ffpp_keypoint_tracking:
    name: "FlowFormer++ Keypoint Tracking Service"
    port: 8001
    type: "fastapi"
    path: "web/ffpp_keypoint_tracking"
    health_endpoint: "/health"
    
  image_labeling:
    name: "Image Labeling Tool"
    port: 8002
    type: "static_web"
    path: "ThirdParty/ImageLabelingWeb"
```

**Note:** Services automatically read ports from config on startup. Edit the YAML file and restart services to apply changes.

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for performance)
- NumPy ≥1.21.0 (supports both 1.x and 2.x)
- Flask (for web services)
- PyYAML (for configuration)
- All dependencies auto-installed with `setup_all_in_one.sh`

## 🎨 Dashboard Features

### Real-Time Monitoring
- **Server-Sent Events (SSE)** for instant updates
- **No polling** - efficient real-time communication
- **Connection status** indicator

### Visual Design
- **16:9 optimized layout** for big screens
- **Dark gradient theme** for comfortable viewing
- **Responsive grid layout** with side-by-side panels
- **Smooth animations** on keypoints (breathing effect)

### Keypoint Visualization
- **Animated markers** - breathing from 1px to 13px
- **Color coding** - Red (original), Green (tracked)
- **Responsive scaling** - keypoints match image size at any resolution
- **Dynamic positioning** - handles window resize

### Data Persistence
- **Automatic image storage** in `output/api_images/`
- **RGB color correction** (fixes BGR→RGB conversion)
- **JSON metadata** for each API call
- **Unlimited history** (counter not capped at 50)

## 🚀 Performance

- **Local tracking:** ~0.3s per frame (with GPU)
- **Web API tracking:** ~0.5s per frame (with GPU + network)
- **21x faster** than external API services (~7s)
- **GPU acceleration** with CUDA
- **NumPy 2.x optimized**

## 📝 Recent Updates

### Version 2.1 (Current - November 2025)
- ✅ **Conda enforcement** - Setup requires Conda by default, with clear installation guidance
- ✅ **Automated Conda installer** - One-command Conda installation with `scripts/install_conda.sh`
- ✅ **Environment naming** - Changed from `flowformerpp` to `robot_vision` for consistency
- ✅ **Simplified validation** - Test script now only runs the main example for faster validation
- ✅ **--skip-conda flag** - Optional system Python support for advanced users
- ✅ **Improved error messages** - Clear guidance when Conda is missing with installation commands
- ✅ **Setup script enhancements** - Better argument parsing and step counting

### Version 2.0
- ✅ Real-time web dashboard with SSE
- ✅ Simplified setup script (60 lines vs 440)
- ✅ Configuration-based port management
- ✅ Fixed RGB color channels in saved images
- ✅ Fixed keypoint scaling with responsive images
- ✅ Unlimited API call counter (removed 50-call limit)
- ✅ Breathing keypoint animation (1px min size)
- ✅ Removed orphaned config files

### Version 1.0
- Initial release with FlowFormer++ integration
- NumPy 2.x compatibility
- Bidirectional validation
- Multiple reference management

## � Troubleshooting

### Conda Not Found

If you see "Conda is not installed!" error:

```bash
# Use our automated installer
bash scripts/install_conda.sh

# Or install manually
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init
source ~/.bashrc
```

After installation, run the setup again:
```bash
bash setup_all_in_one.sh
```

### Setup Fails at a Specific Step

Run individual scripts to debug:

```bash
# Check what's failing
bash scripts/check_requirements.sh
bash scripts/setup_submodules.sh update
bash scripts/setup_conda.sh create
bash scripts/install_dependencies.sh install
bash scripts/download_models.sh download
bash scripts/run_tests.sh
```

### Model Download Issues

If model downloads fail:

```bash
# Check network connectivity
ping google.com

# Try downloading models manually
cd ThirdParty/FlowFormerPlusPlusServer
./scripts/download_ckpts.sh

# Check model status
cd ../..
bash scripts/download_models.sh status
```

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA not available, tracker will use CPU (slower)
# To use CPU explicitly:
tracker = FFPPKeypointTracker(device='cpu')
```

### Using System Python (Not Recommended)

If you absolutely cannot use Conda:

```bash
# Run setup with --skip-conda flag
bash setup_all_in_one.sh --skip-conda

# Note: This may cause dependency conflicts
# Conda is highly recommended for this project
```

### Environment Already Exists

If you see "Conda environment 'robot_vision' already exists":

```bash
# Remove and recreate
conda env remove -n robot_vision
bash setup_all_in_one.sh

# Or use existing environment
conda activate robot_vision
bash scripts/install_dependencies.sh install
```

### Validation Test Fails

If the example fails during setup:

```bash
# Check the error message
conda activate robot_vision
python examples/ffpp_keypoint_tracker_example.py

# Common issues:
# - Missing models: bash scripts/download_models.sh download
# - GPU memory: Reduce image size or use CPU
# - Missing dependencies: bash scripts/install_dependencies.sh install
```

## �📄 License

