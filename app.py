#!/usr/bin/env python3
"""
Robot Vision Keypoint Tracker API Server
========================================

FastAPI-based web service for real keypoint tracking using FlowFormer++.
This version provides production-ready API interface with actual 
FlowFormer++ model integration for real image processing.
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
    title="Robot Vision Keypoint Tracker API (Production)",
    description="High-performance keypoint tracking API with FlowFormer++ integration",
    version="2.0.0-production",
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
async def root():
    """Root endpoint with API documentation."""
    return f"""
    <html>
        <head>
            <title>Robot Vision Keypoint Tracker API (Production)</title>
        </head>
        <body>
            <h1>Robot Vision Keypoint Tracker API - Production</h1>
            <p>🚀 <strong>Real FlowFormer++ Integration</strong></p>
            <p>Status: {'✅ Tracker Ready' if tracker_initialized else '❌ Tracker Not Initialized'}</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/docs">Interactive API Documentation (Swagger)</a></li>
                <li><a href="/redoc">Alternative Documentation (ReDoc)</a></li>
                <li><strong>GET /health</strong> - Health check endpoint</li>
                <li><strong>GET /references</strong> - List stored references</li>
                <li><strong>POST /set_reference</strong> - Set reference image with keypoints</li>
                <li><strong>POST /track_keypoints</strong> - Track keypoints using FlowFormer++</li>
            </ul>
            <p>🔥 Real GPU-accelerated keypoint tracking with FlowFormer++ model!</p>
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
        "service": "Robot Vision Keypoint Tracker API (Production)",
        "status": "healthy" if tracker_initialized else "degraded",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "version": "2.0.0-production",
        "mode": "production_with_real_tracking",
        "tracker_loaded": tracker_initialized,
        "gpu_available": tracker.device.type == 'cuda' if tracker_initialized and tracker.device else False,
        "device": str(tracker.device) if tracker_initialized and tracker.device else "unknown"
    }
    
    return APIResponse(
        success=tracker_initialized,
        message="Service is running with real FlowFormer++ tracking" if tracker_initialized else "Service degraded: tracker not initialized",
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
                    "processing_time": 0.0  # Not provided by the tracker
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
    logger.info("🚀 Starting Robot Vision Keypoint Tracker API...")
    if TRACKER_AVAILABLE:
        success = initialize_tracker()
        if success:
            logger.info("✅ Tracker initialization completed successfully!")
        else:
            logger.warning("⚠️ Tracker initialization failed. Keypoint tracking will be in degraded mode.")
    else:
        logger.warning("⚠️ Tracker not available due to import issues. Keypoint tracking will be in degraded mode.")

if __name__ == "__main__":
    print("🚀 Starting Robot Vision Keypoint Tracker API")
    print("📋 Features:")
    print("   - ✅ Real FlowFormer++ keypoint tracking")
    print("🌐 Access API docs at: http://localhost:8009/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8009,
        reload=False,
        log_level="info"
    )