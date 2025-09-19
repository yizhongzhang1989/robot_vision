#!/usr/bin/env python3
"""
Robot Vision Keypoint Tracker API Server
========================================

FastAPI-based web service for keypoint tracking using FFPPKeypointTracker.
This server provides RESTful APIs for:
- Setting reference images with keypoints
- Tracking keypoints in target images
- Bidirectional flow validation
- Multiple reference image management
- Performance benchmarking

API Endpoints:
- POST /set_reference: Set reference image with keypoints
- POST /track_keypoints: Track keypoints in target image
- GET /references: List all stored reference images
- DELETE /references/{name}: Remove a reference image
- GET /health: Health check endpoint
- GET /: API documentation

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Requirements:
    - FastAPI
    - uvicorn
    - python-multipart (for file uploads)
    - Pillow (for image processing)
    - OpenCV (cv2)
    - numpy

Author: Based on FFPPKeypointTracker example
Date: September 2025
"""

import os
import sys
import json
import time
import logging
import traceback
import warnings
from typing import Dict, List, Optional, Union
from pathlib import Path
import base64
from io import BytesIO

# Suppress matplotlib warnings that don't affect functionality
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Add the current directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import the keypoint tracker with fallback strategies
FFPPKeypointTracker = None
try:
    # Try direct import first
    import importlib.util
    spec = importlib.util.spec_from_file_location("ffpp_keypoint_tracker", 
                                                 os.path.join(current_dir, "core", "ffpp_keypoint_tracker.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    FFPPKeypointTracker = module.FFPPKeypointTracker
except Exception as e1:
    try:
        # Fallback to standard import
        from core.ffpp_keypoint_tracker import FFPPKeypointTracker
    except ImportError as e2:
        logger.error(f"Failed to import FFPPKeypointTracker: {e2}")

if FFPPKeypointTracker is None:
    # Define a placeholder that will be replaced when needed
    class _PlaceholderTracker:
        pass
    
    original_FFPPKeypointTracker = FFPPKeypointTracker
    FFPPKeypointTracker = _PlaceholderTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Robot Vision Keypoint Tracker API",
    description="High-performance keypoint tracking API using FlowFormer++",
    version="1.0.0",
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

# Global tracker instance
tracker: Optional[FFPPKeypointTracker] = None

# Pydantic models for request/response
class KeypointModel(BaseModel):
    x: float = Field(..., description="X coordinate of the keypoint")
    y: float = Field(..., description="Y coordinate of the keypoint")
    id: Optional[int] = Field(None, description="Optional keypoint ID")
    label: Optional[str] = Field(None, description="Optional keypoint label")

class SetReferenceRequest(BaseModel):
    keypoints: List[KeypointModel] = Field(..., description="List of keypoints")
    image_name: Optional[str] = Field(None, description="Optional reference image name")

class TrackKeypointsRequest(BaseModel):
    reference_name: Optional[str] = Field(None, description="Name of reference image to use")
    bidirectional: bool = Field(False, description="Enable bidirectional validation")

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: str

class TrackedKeypointModel(BaseModel):
    x: float
    y: float
    id: Optional[int] = None
    label: Optional[str] = None
    displacement_x: Optional[float] = None
    displacement_y: Optional[float] = None
    consistency_distance: Optional[float] = None

class TrackingResultModel(BaseModel):
    tracked_keypoints: List[TrackedKeypointModel]
    keypoints_count: int
    processing_time: float
    bidirectional_stats: Optional[Dict] = None

# Utility functions
def image_to_numpy(image_file: UploadFile) -> np.ndarray:
    """Convert uploaded image file to numpy array."""
    try:
        # Read image data
        image_data = image_file.file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error converting image to numpy array: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 string for JSON response."""
    try:
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_string
    
    except Exception as e:
        logger.error(f"Error converting numpy array to base64: {e}")
        return ""

def get_tracker():
    """Dependency to get the global tracker instance."""
    tracker_instance = initialize_tracker()
    if tracker_instance is None:
        raise HTTPException(status_code=503, detail="Tracker not initialized")
    if not tracker_instance.model_loaded:
        raise HTTPException(status_code=503, detail="Tracker model not loaded")
    return tracker_instance

# Initialize tracker on first request (lazy loading)
def initialize_tracker():
    """Initialize the keypoint tracker lazily."""
    global tracker, FFPPKeypointTracker
    if tracker is None:
        try:
            # If we have a placeholder, try to import the real class
            if FFPPKeypointTracker == _PlaceholderTracker:
                try:
                    from core.ffpp_keypoint_tracker import FFPPKeypointTracker as RealTracker
                    FFPPKeypointTracker = RealTracker
                except ImportError as e:
                    logger.error(f"Failed to import FFPPKeypointTracker: {e}")
                    raise Exception(f"Cannot import FFPPKeypointTracker: {e}")
            
            start_time = time.time()
            tracker = FFPPKeypointTracker()
            
            if not tracker.model_loaded:
                logger.error("Failed to load FFPPKeypointTracker model")
                raise Exception("Model loading failed")
            
            init_time = time.time() - start_time
            logger.info(f"Tracker initialized in {init_time:.2f}s on {tracker.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            logger.error(traceback.format_exc())
            tracker = None
    return tracker

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    return """
    <html>
        <head>
            <title>Robot Vision Keypoint Tracker API</title>
        </head>
        <body>
            <h1>Robot Vision Keypoint Tracker API</h1>
            <p>High-performance keypoint tracking API using FlowFormer++</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/docs">Interactive API Documentation (Swagger)</a></li>
                <li><a href="/redoc">Alternative Documentation (ReDoc)</a></li>
                <li><strong>POST /set_reference</strong> - Set reference image with keypoints</li>
                <li><strong>POST /track_keypoints</strong> - Track keypoints in target image</li>
                <li><strong>GET /references</strong> - List all stored reference images</li>
                <li><strong>DELETE /references/{name}</strong> - Remove a reference image</li>
                <li><strong>GET /health</strong> - Health check endpoint</li>
            </ul>
            <h2>Usage Examples:</h2>
            <p>See <a href="/docs">interactive documentation</a> for detailed usage examples.</p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global tracker
    
    status = {
        "service": "Robot Vision Keypoint Tracker API",
        "status": "healthy" if tracker and tracker.model_loaded else "not_initialized",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "version": "1.0.0"
    }
    
    if tracker:
        status.update({
            "tracker_loaded": tracker.model_loaded,
            "device": getattr(tracker, 'device', 'unknown'),
            "references_count": len(getattr(tracker, 'reference_data', {}))
        })
    else:
        status.update({
            "tracker_loaded": False,
            "device": "unknown",
            "references_count": 0,
            "note": "Tracker will be initialized on first API call"
        })
    
    return APIResponse(
        success=True,  # Health check always succeeds
        message="Service is running",
        data=status,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    )

@app.post("/set_reference")
async def set_reference_image(
    image: UploadFile = File(..., description="Reference image file (JPEG, PNG, etc.)"),
    keypoints: str = Form(..., description="JSON string of keypoints list"),
    image_name: Optional[str] = Form(None, description="Optional reference image name"),
    tracker_instance: FFPPKeypointTracker = Depends(get_tracker)
):
    """
    Set reference image with keypoints for tracking.
    
    This endpoint allows you to upload a reference image and specify keypoints
    that will be tracked in subsequent target images.
    """
    try:
        start_time = time.time()
        
        # Parse keypoints JSON
        try:
            keypoints_data = json.loads(keypoints)
            # Convert to expected format
            keypoints_list = []
            for kp in keypoints_data:
                if isinstance(kp, dict):
                    keypoints_list.append({
                        'x': float(kp['x']),
                        'y': float(kp['y']),
                        'id': kp.get('id'),
                        'label': kp.get('label')
                    })
                elif isinstance(kp, list) and len(kp) >= 2:
                    keypoints_list.append({
                        'x': float(kp[0]),
                        'y': float(kp[1])
                    })
                else:
                    raise ValueError(f"Invalid keypoint format: {kp}")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid keypoints format: {str(e)}")
        
        # Convert image to numpy array
        image_array = image_to_numpy(image)
        
        # Set reference image
        result = tracker_instance.set_reference_image(
            image_array, 
            keypoints_list, 
            image_name
        )
        
        processing_time = time.time() - start_time
        
        if result['success']:
            logger.info(f"Reference set: {result.get('keypoints_count', 0)} keypoints")
            
            return APIResponse(
                success=True,
                message=f"Reference image set successfully with {result.get('keypoints_count', 0)} keypoints",
                data={
                    "keypoints_count": result.get('keypoints_count', 0),
                    "image_name": result.get('image_name', 'default'),
                    "processing_time": processing_time,
                    "image_shape": image_array.shape[:2]  # Height, Width
                },
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
        else:
            logger.error(f"Failed to set reference image: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=400, detail=result.get('error', 'Failed to set reference image'))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in set_reference_image: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/track_keypoints")
async def track_keypoints(
    image: UploadFile = File(..., description="Target image file to track keypoints in"),
    reference_name: Optional[str] = Form(None, description="Name of reference image to use"),
    bidirectional: bool = Form(False, description="Enable bidirectional validation"),
    tracker_instance: FFPPKeypointTracker = Depends(get_tracker)
):
    """
    Track keypoints from reference image to target image.
    
    This endpoint tracks keypoints from a previously set reference image
    to the uploaded target image using optical flow.
    """
    try:
        start_time = time.time()
        
        # Check if we have any reference images
        if not tracker_instance.reference_data:
            raise HTTPException(
                status_code=400, 
                detail="No reference images available. Use /set_reference endpoint first."
            )
        
        # Convert target image to numpy array
        target_image_array = image_to_numpy(image)
        
        # Track keypoints
        result = tracker_instance.track_keypoints(
            target_image_array,
            reference_name=reference_name,
            bidirectional=bidirectional
        )
        
        processing_time = time.time() - start_time
        
        if result['success']:
            tracked_count = len(result.get('tracked_keypoints', []))
            logger.info(f"Tracked {tracked_count} points in {processing_time:.2f}s")
            
            # Format tracked keypoints
            formatted_keypoints = []
            for kp in result.get('tracked_keypoints', []):
                formatted_kp = {
                    'x': kp['x'],
                    'y': kp['y'],
                    'displacement_x': kp.get('displacement_x', 0),
                    'displacement_y': kp.get('displacement_y', 0)
                }
                
                # Add optional fields if present
                for field in ['id', 'label', 'consistency_distance']:
                    if field in kp:
                        formatted_kp[field] = kp[field]
                
                formatted_keypoints.append(formatted_kp)
            
            response_data = {
                "tracked_keypoints": formatted_keypoints,
                "keypoints_count": tracked_count,
                "processing_time": processing_time,
                "total_processing_time": result.get('total_processing_time', processing_time),
                "reference_used": reference_name or tracker_instance.default_reference_key,
                "bidirectional_enabled": bidirectional,
                "target_image_shape": target_image_array.shape[:2]  # Height, Width
            }
            
            # Add bidirectional statistics if available
            if 'bidirectional_stats' in result:
                response_data['bidirectional_stats'] = result['bidirectional_stats']
            
            # Add processing statistics if available
            if 'processing_stats' in result:
                response_data['processing_stats'] = result['processing_stats']
            
            return APIResponse(
                success=True,
                message=f"Successfully tracked {tracked_count} keypoints",
                data=response_data,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
        else:
            logger.error(f"Keypoint tracking failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=400, detail=result.get('error', 'Keypoint tracking failed'))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in track_keypoints: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/references")
async def list_references(tracker_instance: FFPPKeypointTracker = Depends(get_tracker)):
    """List all stored reference images."""
    try:
        references_info = {}
        
        for name, ref_data in tracker_instance.reference_data.items():
            references_info[name] = {
                "keypoints_count": len(ref_data.get('keypoints', [])),
                "image_shape": ref_data.get('image', np.array([])).shape[:2] if 'image' in ref_data else None,
                "is_default": name == tracker_instance.default_reference_key
            }
        
        return APIResponse(
            success=True,
            message=f"Found {len(references_info)} reference images",
            data={
                "references": references_info,
                "default_reference": tracker_instance.default_reference_key,
                "total_count": len(references_info)
            },
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    except Exception as e:
        logger.error(f"Error listing references: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/references/{reference_name}")
async def remove_reference(
    reference_name: str,
    tracker_instance: FFPPKeypointTracker = Depends(get_tracker)
):
    """Remove a stored reference image."""
    try:
        if reference_name not in tracker_instance.reference_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Reference '{reference_name}' not found. Available: {list(tracker_instance.reference_data.keys())}"
            )
        
        # Remove the reference
        tracker_instance.remove_reference_image(reference_name)
        
        logger.info(f"Removed reference '{reference_name}'")
        
        return APIResponse(
            success=True,
            message=f"Reference '{reference_name}' removed successfully",
            data={
                "removed_reference": reference_name,
                "remaining_references": list(tracker_instance.reference_data.keys()),
                "current_default": tracker_instance.default_reference_key
            },
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing reference: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/benchmark")
async def run_benchmark(
    image: UploadFile = File(..., description="Target image file for benchmarking"),
    reference_name: Optional[str] = Form(None, description="Reference image to use"),
    num_runs: int = Form(5, description="Number of benchmark runs"),
    tracker_instance: FFPPKeypointTracker = Depends(get_tracker)
):
    """
    Run performance benchmark on keypoint tracking.
    
    This endpoint runs multiple tracking operations to measure performance
    statistics including mean, std, min, and max execution times.
    """
    try:
        if not tracker_instance.reference_data:
            raise HTTPException(
                status_code=400,
                detail="No reference images available. Use /set_reference endpoint first."
            )
        
        if num_runs < 1 or num_runs > 20:
            raise HTTPException(status_code=400, detail="num_runs must be between 1 and 20")
        
        # Convert target image to numpy array
        target_image_array = image_to_numpy(image)
        
        logger.info(f"Running benchmark ({num_runs} runs)")
        
        # Run standard tracking benchmark
        standard_times = []
        for i in range(num_runs):
            start_time = time.time()
            result = tracker_instance.track_keypoints(target_image_array, reference_name=reference_name)
            elapsed_time = time.time() - start_time
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=f"Tracking failed on run {i+1}")
            
            standard_times.append(elapsed_time)
        
        # Run bidirectional tracking benchmark
        bidirectional_times = []
        for i in range(num_runs):
            start_time = time.time()
            result = tracker_instance.track_keypoints(
                target_image_array, 
                reference_name=reference_name, 
                bidirectional=True
            )
            elapsed_time = time.time() - start_time
            
            if not result['success']:
                raise HTTPException(status_code=400, detail=f"Bidirectional tracking failed on run {i+1}")
            
            bidirectional_times.append(elapsed_time)
        
        # Calculate statistics
        standard_stats = {
            'mean_time': float(np.mean(standard_times)),
            'std_time': float(np.std(standard_times)),
            'min_time': float(np.min(standard_times)),
            'max_time': float(np.max(standard_times)),
            'keypoints_count': len(result.get('tracked_keypoints', []))
        }
        
        bidirectional_stats = {
            'mean_time': float(np.mean(bidirectional_times)),
            'std_time': float(np.std(bidirectional_times)),
            'min_time': float(np.min(bidirectional_times)),
            'max_time': float(np.max(bidirectional_times)),
            'keypoints_count': len(result.get('tracked_keypoints', []))
        }
        
        # Calculate overhead
        overhead = bidirectional_stats['mean_time'] - standard_stats['mean_time']
        overhead_pct = (overhead / standard_stats['mean_time']) * 100
        
        logger.info(f"Benchmark complete: overhead +{overhead:.2f}s ({overhead_pct:.1f}%)")
        
        return APIResponse(
            success=True,
            message=f"Benchmark completed with {num_runs} runs",
            data={
                "num_runs": num_runs,
                "standard_tracking": standard_stats,
                "bidirectional_tracking": bidirectional_stats,
                "overhead_analysis": {
                    "bidirectional_overhead_seconds": overhead,
                    "bidirectional_overhead_percent": overhead_pct
                },
                "reference_used": reference_name or tracker_instance.default_reference_key
            },
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in benchmark: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    )

# Main entry point
if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        lifespan="off"  # Disable lifespan protocol to avoid warnings
    )