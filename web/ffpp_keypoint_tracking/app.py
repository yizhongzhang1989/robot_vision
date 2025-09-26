#!/usr/bin/env python3
"""
FlowFormer++ Keypoint Tracking Service
=====================================

Dedicated FastAPI service for keypoint tracking using FlowFormer++.
Part of the Robot Vision Services architecture.
"""

import os
import sys
import json
import time
import logging
import traceback
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from PIL import Image
import io

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import the real FlowFormer++ tracker
try:
    from core.ffpp_keypoint_tracker import FFPPKeypointTracker
    TRACKER_AVAILABLE = True
except ImportError as e:
    TRACKER_AVAILABLE = False
    FFPPKeypointTracker = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the real FlowFormer++ tracker
tracker = None
tracker_initialized = False

def initialize_tracker():
    """Initialize the FlowFormer++ tracker."""
    global tracker, tracker_initialized
    
    if not TRACKER_AVAILABLE:
        logger.error("❌ FFPPKeypointTracker is not available due to import errors")
        return False
    
    try:
        logger.info("🚀 Initializing FlowFormer++ Keypoint Tracker...")
        tracker = FFPPKeypointTracker(
            device='auto',  # Auto-detect GPU/CPU
            max_image_size=1024  # Reasonable size for API use
        )
        tracker_initialized = True
        logger.info("✅ FlowFormer++ Tracker initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize tracker: {str(e)}")
        logger.error(traceback.format_exc())
        tracker_initialized = False
        return False

# Initialize FastAPI app
app = FastAPI(
    title="FlowFormer++ Keypoint Tracking Service",
    description="High-performance keypoint tracking service with FlowFormer++ integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: str

class KeypointModel(BaseModel):
    x: float = Field(..., description="X coordinate of the keypoint")
    y: float = Field(..., description="Y coordinate of the keypoint")
    id: Optional[int] = Field(None, description="Optional keypoint ID")
    label: Optional[str] = Field(None, description="Optional keypoint label")

def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file and convert to numpy array."""
    try:
        # Read image file
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        return image_np
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    finally:
        file.file.seek(0)  # Reset file pointer for potential reuse

@app.get("/", response_class=HTMLResponse)
async def service_info():
    """Service information endpoint."""
    return f"""
    <html>
        <head>
            <title>FlowFormer++ Keypoint Tracking Service</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .status {{ color: {'green' if tracker_initialized else 'red'}; font-weight: bold; }}
                .endpoint {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>🎯 FlowFormer++ Keypoint Tracking Service</h1>
            <p class="status">Status: {'✅ Ready' if tracker_initialized else '❌ Not Initialized'}</p>
            <p><strong>FlowFormer++ Integration</strong> - High-performance keypoint tracking</p>
            
            <h2>Available Endpoints:</h2>
            <div class="endpoint"><strong>GET /health</strong> - Service health check</div>
            <div class="endpoint"><strong>GET /references</strong> - List stored reference images</div>
            <div class="endpoint"><strong>POST /set_reference</strong> - Set reference image with keypoints</div>
            <div class="endpoint"><strong>POST /track_keypoints</strong> - Track keypoints using FlowFormer++</div>
            
            <h2>Documentation:</h2>
            <p><a href="/docs">📖 Interactive API Documentation (Swagger)</a></p>
            <p><a href="/redoc">📚 Alternative Documentation (ReDoc)</a></p>
            
            <h2>Service Info:</h2>
            <p>🚀 GPU-accelerated keypoint tracking with FlowFormer++ model</p>
            <p>🔧 Part of Robot Vision Services Architecture</p>
            <p><a href="http://localhost:8000">🏠 Return to Control Center</a></p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Try to initialize tracker if not already done
    if not tracker_initialized:
        initialize_tracker()
    
    status = {
        "service": "FlowFormer++ Keypoint Tracking Service",
        "status": "healthy" if tracker_initialized else "degraded",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "version": "2.0.0",
        "port": 8001,
        "tracker_loaded": tracker_initialized,
        "gpu_available": tracker.device.type == 'cuda' if tracker_initialized and tracker.device else False,
        "device": str(tracker.device) if tracker_initialized and tracker.device else "unknown"
    }
    
    return APIResponse(
        success=tracker_initialized,
        message="Service is running with FlowFormer++ tracking" if tracker_initialized else "Service degraded: tracker not initialized",
        data=status,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    )

@app.get("/references")
async def list_references():
    """List all stored reference images."""
    if not tracker_initialized:
        raise HTTPException(status_code=503, detail="Tracker not initialized")
    
    references = {}
    if hasattr(tracker, 'reference_data'):
        for ref_name, ref_data in tracker.reference_data.items():
            references[ref_name] = {
                "keypoints_count": len(ref_data.processed_keypoints if hasattr(ref_data, 'processed_keypoints') else []),
                "image_shape": ref_data.processed_size if hasattr(ref_data, 'processed_size') else 'unknown',
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    return APIResponse(
        success=True,
        message=f"Found {len(references)} reference images",
        data={
            "references": references,
            "total_count": len(references),
            "default_reference": tracker.default_reference_key if tracker_initialized else None
        },
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    )

@app.post("/set_reference")
async def set_reference_image(
    image: UploadFile = File(..., description="Reference image file"),
    keypoints: str = Form(..., description="JSON string of keypoints"),
    image_name: Optional[str] = Form(None, description="Optional reference image name")
):
    """Set reference image with keypoints using real FlowFormer++ tracker."""
    if not tracker_initialized:
        raise HTTPException(status_code=503, detail="Tracker not initialized. Check /health endpoint.")
    
    try:
        # Load image
        image_np = load_image_from_upload(image)
        
        # Parse keypoints
        keypoints_data = json.loads(keypoints)
        
        # Validate keypoints format
        for kp in keypoints_data:
            if not isinstance(kp, dict) or 'x' not in kp or 'y' not in kp:
                raise ValueError("Each keypoint must be a dict with 'x' and 'y' keys")
        
        # Use tracker to set reference image
        result = tracker.set_reference_image(
            image=image_np,
            keypoints=keypoints_data,
            image_name=image_name
        )
        
        if result.get('success', False):
            return APIResponse(
                success=True,
                message=f"Reference image set successfully with {len(keypoints_data)} keypoints",
                data={
                    "keypoints_count": len(keypoints_data),
                    "image_name": result.get('key'),
                    "image_shape": result.get('regularized_image_shape'),
                    "processing_time": 0.0
                },
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to set reference image: {result.get('error', 'unknown error')}"
            )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid keypoints JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error in set_reference: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/track_keypoints")
async def track_keypoints(
    image: UploadFile = File(..., description="Target image file"),
    reference_name: Optional[str] = Form(None, description="Reference image name"),
    bidirectional: bool = Form(False, description="Enable bidirectional validation")
):
    """Track keypoints using real FlowFormer++ model."""
    if not tracker_initialized:
        raise HTTPException(status_code=503, detail="Tracker not initialized. Check /health endpoint.")
    
    # Check if we have any reference images
    if not hasattr(tracker, 'reference_data') or not tracker.reference_data:
        raise HTTPException(
            status_code=400,
            detail="No reference images available. Use /set_reference endpoint first."
        )
    
    try:
        # Load target image
        target_image_np = load_image_from_upload(image)
        
        # Use tracker to track keypoints
        result = tracker.track_keypoints(
            target_image=target_image_np,
            reference_name=reference_name,
            bidirectional=bidirectional
        )
        
        if result.get('success', False):
            return APIResponse(
                success=True,
                message=f"Keypoint tracking completed successfully",
                data={
                    "tracked_keypoints": result.get('tracked_keypoints', []),
                    "keypoints_count": len(result.get('tracked_keypoints', [])),
                    "processing_time": result.get('total_processing_time', 0),
                    "reference_used": result.get('reference_name'),
                    "bidirectional_enabled": bidirectional,
                    "bidirectional_stats": result.get('bidirectional_stats') if bidirectional else None,
                    "device_used": str(tracker.device) if tracker.device else "unknown"
                },
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Tracking failed: {result.get('error', 'unknown error')}"
            )
        
    except Exception as e:
        logger.error(f"Error in track_keypoints: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the tracker when the app starts."""
    logger.info("🚀 Starting FlowFormer++ Keypoint Tracking Service...")
    if TRACKER_AVAILABLE:
        success = initialize_tracker()
        if success:
            logger.info("✅ FlowFormer++ Keypoint Tracking Service ready!")
        else:
            logger.warning("⚠️ Tracker initialization failed. Service in degraded mode.")
    else:
        logger.warning("⚠️ Tracker not available due to import issues.")

if __name__ == "__main__":
    print("🎯 Starting FlowFormer++ Keypoint Tracking Service")
    print("📋 Features:")
    print("   - ✅ FlowFormer++ keypoint tracking")
    print("   - 🚀 GPU acceleration")  
    print("   - 🔧 Part of Robot Vision Services")
    print("🌐 Access at: http://localhost:8001")
    print("📖 API docs: http://localhost:8001/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )