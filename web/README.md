# Robot Vision Web Services

This direct### Services

1. **Gateway (port 8000)**: Main dashboard with service discovery and health monitoring
2. **FlowFormer++ Keypoint Tracking (port 8001)**: FlowFormer++ based keypoint tracking service
3. **Image Labeling (port 8003)**: Image annotation and labeling interfaceroservices-based web architecture for Robot Vision Services.

## Architecture Overview

The robot vision functionality has been restructured into independent microservices:

```
web/
├── gateway/               # Main control center (port 8000)
├── ffpp_keypoint_tracking/ # FlowFormer++ tracking service (port 8001)
├── image_labeling/        # Image annotation service (port 8003)
└── shared/               # Common web assets
```

## Quick Start

### Option 1: All Services
```bash
python start_services.py
```

### Option 2: Service Manager
```bash
python scripts/manage_services.py start --wait
```

### Option 3: Individual Services
```bash
# Gateway (Control Center)
python web/gateway/app.py

# FlowFormer++ Keypoint Tracking Service  
python web/ffpp_keypoint_tracking/app.py

# Image Labeling Service
python web/image_labeling/app.py
```

## Service Details

### 🤖 Gateway (Control Center) - Port 8000
- **Purpose**: Central control interface and service discovery
- **Features**: Service health monitoring, unified dashboard
- **URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 🎯 FlowFormer++ Keypoint Tracking Service - Port 8001  
- **Purpose**: FlowFormer++ based keypoint tracking
- **Features**: GPU acceleration, bidirectional validation
- **URL**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

### 🏷️ Image Labeling Tool - Port 8003
- **Purpose**: Interactive image annotation and keypoint labeling web interface  
- **Features**: Sub-pixel precision keypoints, zoom/pan, JSON export/import, drag-and-drop
- **Technology**: Existing ImageLabelingWeb (reused from ThirdParty)
- **URL**: http://localhost:8003
- **Type**: Static web interface (no API docs)

## Service Management

### List Services
```bash
python scripts/manage_services.py list
```

### Start All Services
```bash
python scripts/manage_services.py start --wait
```

### Start Individual Service
```bash
python scripts/manage_services.py start-service --service ffpp_keypoint_tracking
```

### Stop All Services
```bash
python scripts/manage_services.py stop
```

## Configuration

Services are configured via `config/services.yaml`:
- Service ports and descriptions
- Dependencies and health endpoints
- Gateway settings

## Development

Services can be either:
- **FastAPI applications**: New microservices with API endpoints
- **Existing web interfaces**: Reused from ThirdParty directory (e.g., ImageLabelingWeb)

The gateway provides service discovery and unified access for both types.

## Migration from Legacy app.py

The original `app.py` has been restructured:
- **Legacy app.py**: Now shows migration information
- **New location**: `web/ffpp_keypoint_tracking/app.py`
- **Enhanced**: Part of microservices architecture with gateway

## Benefits

- ✅ **Service Independence**: Each service runs independently  
- ✅ **Scalability**: Easy to add new vision services
- ✅ **Maintainability**: Clear separation of concerns
- ✅ **Development**: Teams can work on services independently  
- ✅ **Deployment**: Services can be deployed separately or together