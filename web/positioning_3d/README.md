# 3D Positioning Service

Multi-view triangulation service for real-time 3D positioning from robot cameras using FlowFormer++ keypoint tracking.

## Overview

This service provides real-time 3D point reconstruction from multiple camera views captured by robots. It integrates with the FlowFormer++ Keypoint Tracking Service to detect and track feature points across views, then performs multi-view triangulation to compute 3D coordinates.

## Features

- ✅ **Real-time Processing**: Asynchronous image reception with immediate queuing
- ✅ **Multi-Robot Support**: Handle requests from multiple robots concurrently
- ✅ **Serialized FFPP Processing**: Queue-based system prevents concurrent FFPP requests
- ✅ **Auto Reference Upload**: Automatically uploads calibration references at startup
- ✅ **Live Dashboard**: Real-time monitoring with Server-Sent Events (SSE)
- ✅ **Remote FFPP Support**: Can connect to FFPP server on different machines
- ✅ **Robust Error Handling**: Retry logic, timeouts, and graceful degradation

## Architecture

```
[Robot] → [upload_view] → [Task Queue] → [FFPP Worker] → [FFPP Server]
                ↓                            ↓
          [Session Manager]         [Keypoint Tracking]
                ↓                            ↓
         [View Storage]              [Update View]
                ↓                            ↓
    [Check All Tracked?] ← ← ← ← ← ← ← ← ← ┘
                ↓ Yes
         [Triangulation]
                ↓
         [Store Result]
                ↓
      [Broadcast Complete]
```

## Quick Start

### 1. Prepare Dataset

Create reference images in the dataset directory:

```
output/dataset/
    checkerboard_11x8/
        ref_img_1.jpg
        ref_img_1.json
    aruco_marker_6x6/
        ref_img_1.jpg
        ref_img_1.json
```

**JSON format** (`ref_img_1.json`):
```json
{
  "keypoints": [
    {"x": 123.45, "y": 234.56},
    {"x": 234.56, "y": 345.67},
    ...
  ]
}
```

### 2. Start FFPP Service First

```bash
python web/ffpp_keypoint_tracking/app.py
```

Or use a remote FFPP server (e.g., `http://msraig-ubuntu-3:8001`)

### 3. Configure Service (Optional)

Edit `web/positioning_3d/config.yaml` or use CLI arguments:

```yaml
ffpp_server:
  host: "msraig-ubuntu-3"  # Default remote host
  port: 8001

dataset:
  path: "output/dataset"  # Default dataset path
  auto_upload_on_start: true
```

### 4. Start Positioning Service

**Basic usage (uses defaults from config):**
```bash
python web/positioning_3d/app.py
```

**With custom FFPP server:**
```bash
python web/positioning_3d/app.py --ffpp-url http://msraig-ubuntu-3:8001
```

**With custom dataset path:**
```bash
python web/positioning_3d/app.py --dataset-path C:/data/my_references
```

**Full example:**
```bash
python web/positioning_3d/app.py \
    --ffpp-url http://msraig-ubuntu-3:8001 \
    --dataset-path output/dataset \
    --port 8004
```

### 5. Upload References via Dashboard

1. Open http://localhost:8004 in your browser
2. Click **"📤 Upload References to FFPP"** button
3. Wait for confirmation (references will be uploaded to FFPP server)

References are also auto-uploaded on startup if `auto_upload_on_start: true` in config.

## API Usage

### Initialize Session

```python
import requests

response = requests.post('http://localhost:8004/init_session', json={
    'robot_id': 'robot_arm_01',
    'reference_name': 'checkerboard_11x8',
    'num_expected_views': 6
})

session_id = response.json()['session_id']
```

### Upload View

```python
import cv2
import base64
import numpy as np

# Capture image
image = cv2.imread('view_0.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to base64
_, buffer = cv2.imencode('.png', image_rgb)
image_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode()}"

# Upload view
response = requests.post('http://localhost:8004/upload_view', json={
    'session_id': session_id,
    'view_id': 'view_0',
    'image_base64': image_base64,
    'camera_params': {
        'intrinsic': intrinsic_matrix.tolist(),
        'extrinsic': extrinsic_matrix.tolist(),
        'distortion': distortion_coeffs.tolist(),
        'image_size': [width, height]
    }
})

# Returns immediately with queue position
print(f"View queued at position: {response.json()['queue_position']}")
```

### Check Status

```python
response = requests.get(f'http://localhost:8004/session_status/{session_id}')
status = response.json()

print(f"Progress: {status['session']['progress']['views_tracked']}/{status['session']['progress']['expected_views']}")
print(f"Status: {status['session']['status']}")
```

### Get Result

```python
response = requests.get(f'http://localhost:8004/result/{session_id}')
result = response.json()

if result['success']:
    points_3d = np.array(result['result']['points_3d'])
    mean_error = result['result']['mean_error']
    print(f"Triangulated {len(points_3d)} points with {mean_error:.3f}px error")
```

## Real-Time Monitoring

Connect to SSE endpoint for live updates:

```javascript
const evtSource = new EventSource('http://localhost:8004/events');

evtSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Event:', data.type, data.data);
};
```

Event types:
- `session_created` - New session initialized
- `view_uploaded` - View received and queued
- `view_tracked` - Keypoints tracked for view
- `triangulation_started` - Triangulation in progress
- `triangulation_completed` - 3D points ready
- `triangulation_failed` - Triangulation error

## Configuration

### Service Settings

```yaml
service:
  port: 8004
  host: "0.0.0.0"
```

### FFPP Server

```yaml
ffpp_server:
  host: "192.168.1.100"  # Can be remote
  port: 8001
  timeout: 30
  retry_attempts: 3
```

### Queue Management

```yaml
queue:
  max_size: 100
  worker_threads: 1  # Keep at 1 for serialization
```

### Session Management

```yaml
session:
  timeout_minutes: 30
  max_concurrent_sessions: 50
  cleanup_interval_seconds: 300
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/health` | GET | Health check |
| `/init_session` | POST | Create new session |
| `/upload_view` | POST | Upload camera view |
| `/session_status/<id>` | GET | Get session status |
| `/result/<id>` | GET | Get triangulation result |
| `/queue_status` | GET | Get queue statistics |
| `/list_sessions` | GET | List all sessions |
| `/list_references` | GET | List reference images |
| `/events` | GET | SSE event stream |

## Error Handling

- **Queue Full**: Returns 503, client should retry
- **Session Not Found**: Returns 404
- **FFPP Server Down**: Retries with exponential backoff
- **Tracking Failure**: Marks view as failed, continues with remaining views
- **Session Timeout**: Auto-cleanup after 30 minutes of inactivity

## Performance

- **Asynchronous Upload**: Non-blocking view reception
- **Single FFPP Worker**: Prevents server overload
- **Session Isolation**: Each robot's work is independent
- **Real-time Updates**: SSE for instant status changes

## Troubleshooting

### FFPP Server Not Connected

```bash
# Check FFPP service is running
curl http://localhost:8001/health

# Check network connectivity
ping <ffpp_host>
```

### References Not Loading

```bash
# Check dataset path exists
ls ThirdParty/camera_calibration_toolkit/data/camera_calibration_v1

# Check JSON files are present
find ThirdParty/camera_calibration_toolkit/data -name "*.json"
```

### Queue Filling Up

- Check FFPP server performance
- Reduce number of concurrent robots
- Increase queue size in config

## Development

### Running Tests

```bash
# TODO: Add tests
python -m pytest web/positioning_3d/tests/
```

### Adding New Features

1. Update `models.py` for new data structures
2. Add API endpoints in `app.py`
3. Update dashboard template
4. Test with multiple robots

## License

Part of Robot Vision Services - Internal Use

## Author

Yizhong Zhang - November 2025
