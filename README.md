# Robot Vision

High-performanc### Web API Client
```python
from core.ffpp_webapi_keypoint_tracker import FFPPWebAPIKeypointTracker

tracker = FFPPKeypointTracker(api_url='http://localhost:8001')nt tracking for robotics applications with real-time web monitoring dashboard.

## 🚀 Quick Start

```bash
# One-click setup (recommended)
bash setup_all_in_one.sh

# Activate the environment
conda activate flowformerpp

# Start all services
python start_services.py

# Access the dashboard
# Gateway: http://localhost:8000
# FlowFormer++ Tracking: http://localhost:8001
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

The setup script runs 6 steps sequentially:
1. Check system requirements
2. Setup Git submodules
3. Create Conda environment
4. Install dependencies
5. Download models
6. Run tests

**To skip any step**, simply comment it out in `setup_all_in_one.sh`:
```bash
# Step 5: Download models
# echo ""
# echo "Step 5/6: Downloading models..."
# bash "$SCRIPTS_DIR/download_models.sh" download
```

### Option 2: Manual Setup
```bash
# Clone with submodules
git clone --recursive https://github.com/yizhongzhang1989/robot_vision.git
cd robot_vision

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download models
cd ThirdParty/FlowFormerPlusPlusServer
./scripts/download_ckpts.sh
```

## 🌐 Web Services

### Starting Services

```bash
# Activate the conda environment
conda activate flowformerpp

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

### Version 2.0 (Current)
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

## 📄 License
