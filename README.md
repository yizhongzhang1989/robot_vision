# Robot Vision

A robotics vision toolkit for high-performance optical flow–based keypoint tracking and 3D positioning. Combines FlowFormer++ deep learning models with multi-view triangulation, web APIs, and interactive annotation tools.

**Key capabilities:**
- Keypoint tracking via FlowFormer++ (~0.3s/frame on RTX 3090)
- Multi-view 3D triangulation from calibrated cameras
- REST APIs for remote access from any language
- Web dashboard for service monitoring
- Interactive image labeling tool

## Requirements

### Hardware
| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA with CUDA support (tested: RTX 3090, RTX 4090) |
| VRAM | 8 GB+ recommended |
| Disk | ~10 GB free (models + dependencies) |

### Software
| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 22.04 / 24.04 |
| Python | 3.10+ |
| NVIDIA Driver | 530+ (check with `nvidia-smi`) |
| Git | With submodule support |
| Conda | Anaconda or Miniconda (**recommended**) |

### GPU Driver Setup

```bash
# Check if GPU is detected
lspci | grep -i nvidia

# Check if driver is working
nvidia-smi
```

If `nvidia-smi` fails, install the driver:
```bash
sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

## Quick Start

### Option A: Automated Setup (Recommended)

```bash
git clone --recurse-submodules <repo-url>
cd robot_vision

# Full setup: checks requirements, inits submodules, creates conda env,
# installs dependencies, downloads models (~2GB), runs validation tests
bash setup_all_in_one.sh

# Activate environment and start services
conda activate robot_vision
python start_services.py
```

### Option B: Without Conda

If conda is not available, use `--skip-conda` to install with system pip.
Dependencies install to `~/.local/` — you must ensure `~/.local/bin` is on PATH.

```bash
export PATH="$HOME/.local/bin:$PATH"
bash setup_all_in_one.sh --skip-conda
python start_services.py
```

### Option C: Step-by-Step Manual Setup

Run each step individually for more control or to debug issues:

```bash
# 1. Check system requirements
bash scripts/check_requirements.sh            # with conda
bash scripts/check_requirements.sh --skip-conda  # without conda

# 2. Initialize git submodules
bash scripts/setup_submodules.sh update

# 3. Create conda environment (skip if using --skip-conda)
bash scripts/setup_conda.sh create

# 4. Install Python dependencies
bash scripts/install_dependencies.sh install            # with conda
bash scripts/install_dependencies.sh install --skip-conda  # without conda

# 5. Download FlowFormer++ model checkpoints (~310MB total)
bash scripts/download_models.sh download            # with conda
bash scripts/download_models.sh download --skip-conda  # without conda

# 6. Validate installation (runs FFPP example end-to-end)
bash scripts/run_tests.sh all                # with conda
bash scripts/run_tests.sh all --skip-conda   # without conda
```

### Verify Installation

```bash
# Run the full test suite (loads models, tracks keypoints, saves visualizations)
# First run takes ~3 minutes (model backbone download), subsequent runs ~15 seconds
bash scripts/run_tests.sh

# Or run directly
python examples/ffpp_keypoint_tracker_example.py

# Check CUDA devices only
python examples/ffpp_keypoint_tracker_example.py --devices
```

Expected output: `Tests passed: 6/6` with results in `output/ffpp_keypoint_tracker_example_output/`.

## Web Services

### Starting Services

```bash
python start_services.py
```

This starts 4 services:

| Service | Port | Description |
|---------|------|-------------|
| **Gateway** | 8000 | Control dashboard, service discovery, health monitoring |
| **FFPP Tracking** | 8001 | FlowFormer++ keypoint tracking REST API |
| **Image Labeling** | 8002 | Interactive keypoint annotation tool |
| **3D Positioning** | 8004 | Multi-view triangulation with session management |

**Note:** The FFPP Tracking service loads the FlowFormer++ model on startup, which takes ~3 minutes on first run (downloads backbone weights) and ~10 seconds on subsequent runs. The 3D Positioning service connects to FFPP; if FFPP is still loading, it starts in degraded mode and reconnects automatically.

### Checking Service Health

```bash
# All services at once (via gateway)
curl http://localhost:8000/services/status

# Individual services
curl http://localhost:8001/health   # FFPP Tracking
curl http://localhost:8004/health   # 3D Positioning
```

### Service Configuration

- **Service ports and settings:** `config/services.yaml`
- **3D Positioning config** (FFPP host, session timeouts, queue): `web/positioning_3d/config.yaml`

### Stopping Services

Press `Ctrl+C` in the terminal running `start_services.py`, or:
```bash
# Kill all service ports
fuser -k 8000/tcp 8001/tcp 8002/tcp 8004/tcp
```

## Usage

### Python API (Direct)

Best performance — model runs in-process on GPU.

```python
from core.ffpp_keypoint_tracker import FFPPKeypointTracker

# Initialize (loads model onto GPU)
tracker = FFPPKeypointTracker()

# Set reference image with keypoints
tracker.set_reference_image(ref_image, keypoints)  # RGB numpy array, list of {'x', 'y'}

# Track keypoints in a new image
result = tracker.track_keypoints(target_image)
# result['tracked_keypoints'] = [{'x': ..., 'y': ...}, ...]

# With bidirectional validation for accuracy assessment
result = tracker.track_keypoints(target_image, bidirectional=True)
# result['consistency_distances'] = [float, ...]  (lower = more accurate)
```

### Web API (HTTP)

Same interface, works over the network. Requires services running.

```python
from core.ffpp_webapi_keypoint_tracker import FFPPWebAPIKeypointTracker

tracker = FFPPWebAPIKeypointTracker(api_url='http://localhost:8001')
tracker.set_reference_image(ref_image, keypoints)
result = tracker.track_keypoints(target_image)
```

### 3D Triangulation

```python
from core.triangulation import triangulate_multiview

# points_2d: list of N views, each a list of M 2D points (or None if not visible)
# projection_matrices: list of N 3x4 projection matrices
points_3d, errors = triangulate_multiview(points_2d, projection_matrices)
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `http://localhost:8000/` | GET | Gateway dashboard |
| `http://localhost:8000/services/status` | GET | Health of all services |
| `http://localhost:8001/health` | GET | FFPP service health |
| `http://localhost:8001/set_reference_image` | POST | Upload reference image + keypoints |
| `http://localhost:8001/track_keypoints` | POST | Track keypoints in target image |
| `http://localhost:8004/health` | GET | 3D Positioning health |
| `http://localhost:8004/sessions` | POST | Create triangulation session |

## Project Structure

```
robot_vision/
├── core/                          # Python library modules
│   ├── ffpp_keypoint_tracker.py   # Direct FlowFormer++ tracker (GPU)
│   ├── ffpp_webapi_keypoint_tracker.py  # HTTP client wrapper
│   ├── keypoint_tracker.py        # Abstract base class
│   ├── triangulation.py           # Multi-view 3D triangulation (DLT)
│   ├── 3d_coordinates_estimation.py  # File-based 3D estimation
│   ├── positioning_3d_webapi.py   # 3D service client
│   └── utils.py                   # Keypoint utilities, visualization
├── web/                           # Web service applications
│   ├── gateway/                   # Service discovery & dashboard (Flask, port 8000)
│   ├── ffpp_keypoint_tracking/    # Tracking REST API (Flask, port 8001)
│   ├── positioning_3d/            # 3D triangulation service (Flask, port 8004)
│   ├── image_labeling/            # Annotation tool wrapper (port 8002)
│   └── shared/                    # Shared CSS
├── examples/                      # Usage examples and test scripts
├── config/                        # Service configuration (services.yaml)
├── scripts/                       # Setup and management scripts
├── sample_data/                   # Test images and keypoints
├── ThirdParty/                    # Git submodules
│   ├── FlowFormerPlusPlusServer/  # FlowFormer++ model and inference
│   ├── camera_calibration_toolkit/  # Camera calibration tools
│   └── ImageLabelingWeb/          # Web-based image labeling UI
├── setup_all_in_one.sh            # One-command setup script
├── start_services.py              # Start all web services
└── requirements.txt               # Python dependencies
```

## Examples

| Example | Description |
|---------|-------------|
| `examples/ffpp_keypoint_tracker_example.py` | Direct API: tracking, bidirectional validation, multiple references, flow visualization, benchmarks |
| `examples/ffpp_webapi_keypoint_tracker_example.py` | Web API client with configurable image encoding |
| `examples/triangulation_example.py` | Multi-view 3D triangulation from calibrated cameras |
| `examples/estimate_3d_coordinates_example.py` | File-based 3D coordinate estimation CLI |
| `examples/fitting_example.py` | Rigid transformation fitting between point sets |
| `examples/positioning_3d_webapi_example.py` | 3D Positioning service session workflow |

## Troubleshooting

### `nvidia-smi` not found or fails
Install the NVIDIA driver: `sudo ubuntu-drivers autoinstall && sudo reboot`

### `torch.cuda.is_available()` returns False
PyTorch may have been installed without CUDA support. Reinstall:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Port already in use when starting services
Kill leftover processes:
```bash
fuser -k 8000/tcp 8001/tcp 8002/tcp 8004/tcp
```

### `gdown` or `uvicorn` command not found (--skip-conda mode)
pip installs scripts to `~/.local/bin`. Add to PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```
Add this line to `~/.bashrc` to make it permanent.

### matplotlib Axes3D warning
Harmless warning caused by system `python3-matplotlib` conflicting with pip-installed version. Does not affect functionality. Using conda avoids this.

### 3D Positioning shows "FFPP server not available" on startup
Expected on first start — the FFPP service takes ~3 minutes to load models. The 3D service starts in degraded mode and reconnects automatically once FFPP is ready. Check with:
```bash
curl http://localhost:8004/health  # ffpp_server.connected should be true
```

### Model download fails
Re-run the download script:
```bash
bash scripts/download_models.sh download --skip-conda  # or without --skip-conda
# Or manually:
cd ThirdParty/FlowFormerPlusPlusServer && bash scripts/download_ckpts.sh
```
