#!/usr/bin/env python3
"""
FlowFormer++ Keypoint Tracking Service
=====================================

Dedicated Flask service for keypoint tracking using FlowFormer++.
Part of the Robot Vision Services architecture.
"""

import os
import sys
import json
import time
import logging
import traceback
import base64
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
from werkzeug.datastructures import FileStorage
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

# Initialize Flask app
app = Flask(__name__)

# Helper function to create API responses
def create_api_response(success: bool, message: str, data: Optional[Dict] = None, error: Optional[str] = None) -> Dict:
    return {
        'success': success,
        'message': message,
        'data': data,
        'error': error,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode base64 encoded JPG image and convert to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',', 1)[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)
        
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        return image_np
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")

def encode_image_to_base64(image_np: np.ndarray, quality: int = 85) -> str:
    """Encode numpy image array to base64 JPG string."""
    try:
        # Ensure image is in the correct format
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_np, 'RGB')
        
        # Convert to JPG bytes with compression
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG', quality=quality)
        img_bytes.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return image_base64
    except Exception as e:
        raise ValueError(f"Failed to encode image to base64: {str(e)}")

def load_image_from_upload(file: FileStorage) -> np.ndarray:
    """Load image from uploaded file and convert to numpy array."""
    try:
        # Read image file
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        return image_np
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")
    finally:
        file.seek(0)  # Reset file pointer for potential reuse

@app.route("/")
def service_info():
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
            
            <h2>Tracker-Compatible Endpoints:</h2>
            <div class="endpoint"><strong>POST /set_reference_image</strong> - Set reference image with keypoints</div>
            <div class="endpoint"><strong>POST /track_keypoints</strong> - Track keypoints using FlowFormer++</div>
            <div class="endpoint"><strong>POST /remove_reference_image</strong> - Remove reference image by name</div>
            
            <h2>Service Endpoints:</h2>
            <div class="endpoint"><strong>GET /health</strong> - Service health check</div>
            <div class="endpoint"><strong>GET /references</strong> - List stored reference images</div>
            
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

@app.route("/health")
def health_check():
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
    
    return jsonify(create_api_response(
        success=tracker_initialized,
        message="Service is running with FlowFormer++ tracking" if tracker_initialized else "Service degraded: tracker not initialized",
        data=status
    ))

@app.route("/references")
def list_references():
    """List all stored reference images."""
    if not tracker_initialized:
        return jsonify(create_api_response(
            success=False,
            message="Tracker not initialized"
        )), 503
    
    references = {}
    if hasattr(tracker, 'reference_data'):
        for ref_name, ref_data in tracker.reference_data.items():
            references[ref_name] = {
                "keypoints_count": len(ref_data.processed_keypoints if hasattr(ref_data, 'processed_keypoints') else []),
                "image_shape": ref_data.processed_size if hasattr(ref_data, 'processed_size') else 'unknown',
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    return jsonify(create_api_response(
        success=True,
        message=f"Found {len(references)} reference images",
        data={
            "references": references,
            "total_count": len(references),
            "default_reference": tracker.default_reference_key if tracker_initialized else None
        }
    ))

# ============================================================================
# TRACKER-COMPATIBLE ENDPOINTS
# These provide the exact same method names as the tracker interface
# ============================================================================

@app.route("/set_reference_image", methods=["POST"])
def set_reference_image():
    """Set reference image with keypoints using real FlowFormer++ tracker."""
    if not tracker_initialized:
        return jsonify(create_api_response(
            success=False,
            message="Tracker not initialized. Check /health endpoint."
        )), 503
    
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify(create_api_response(
                success=False,
                message="Request must be JSON with base64 encoded image"
            )), 400
        
        data = request.get_json()
        
        # Check required fields
        if 'image_base64' not in data:
            return jsonify(create_api_response(
                success=False,
                message="No image_base64 field provided"
            )), 400
        
        if 'keypoints' not in data:
            return jsonify(create_api_response(
                success=False,
                message="No keypoints field provided"
            )), 400
        
        # Get data fields
        image_base64 = data['image_base64']
        keypoints_data = data['keypoints']
        image_name = data.get('image_name')
        
        # Decode and load image
        image_np = decode_base64_image(image_base64)
        
        # Validate keypoints format
        if not isinstance(keypoints_data, list):
            return jsonify(create_api_response(
                success=False,
                message="Keypoints must be a list"
            )), 400
        
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
            return jsonify(create_api_response(
                success=True,
                message=f"Reference image set successfully with {len(keypoints_data)} keypoints",
                data={
                    "keypoints_count": len(keypoints_data),
                    "image_name": result.get('key'),
                    "image_shape": result.get('regularized_image_shape'),
                    "processing_time": 0.0
                }
            ))
        else:
            return jsonify(create_api_response(
                success=False,
                message=f"Failed to set reference image: {result.get('error', 'unknown error')}"
            )), 500
        
    except json.JSONDecodeError as e:
        return jsonify(create_api_response(
            success=False,
            message=f"Invalid keypoints JSON: {str(e)}"
        )), 400
    except Exception as e:
        logger.error(f"Error in set_reference_image: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify(create_api_response(
            success=False,
            message=f"Error: {str(e)}"
        )), 500

@app.route("/track_keypoints", methods=["POST"])
def track_keypoints():
    """Track keypoints using real FlowFormer++ model."""
    if not tracker_initialized:
        return jsonify(create_api_response(
            success=False,
            message="Tracker not initialized. Check /health endpoint."
        )), 503
    
    # Check if we have any reference images
    if not hasattr(tracker, 'reference_data') or not tracker.reference_data:
        return jsonify(create_api_response(
            success=False,
            message="No reference images available. Use /set_reference_image endpoint first."
        )), 400
    
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify(create_api_response(
                success=False,
                message="Request must be JSON with base64 encoded image"
            )), 400
        
        data = request.get_json()
        
        # Check required fields
        if 'image_base64' not in data:
            return jsonify(create_api_response(
                success=False,
                message="No image_base64 field provided"
            )), 400
        
        # Get data fields
        image_base64 = data['image_base64']
        reference_name = data.get('reference_name')
        bidirectional = data.get('bidirectional', False)
        
        # Decode target image
        target_image_np = decode_base64_image(image_base64)
        
        # Use tracker to track keypoints
        result = tracker.track_keypoints(
            target_image=target_image_np,
            reference_name=reference_name,
            bidirectional=bidirectional
        )
        
        if result.get('success', False):
            return jsonify(create_api_response(
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
                }
            ))
        else:
            return jsonify(create_api_response(
                success=False,
                message=f"Tracking failed: {result.get('error', 'unknown error')}"
            )), 500
        
    except Exception as e:
        logger.error(f"Error in track_keypoints: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify(create_api_response(
            success=False,
            message=f"Error: {str(e)}"
        )), 500

@app.route("/remove_reference_image", methods=["POST", "DELETE"])
def remove_reference_image():
    """Remove a stored reference image by name.
    
    This endpoint provides server-side reference removal to match the tracker interface.
    Accepts both POST and DELETE methods for flexibility.
    """
    if not tracker_initialized:
        return jsonify(create_api_response(
            success=False,
            message="Tracker not initialized. Check /health endpoint."
        )), 503
    
    try:
        # Handle both JSON and query parameters
        image_name = None
        
        if request.method == "POST" and request.is_json:
            data = request.get_json()
            image_name = data.get('image_name')
        elif request.method == "DELETE":
            image_name = request.args.get('image_name')
        else:
            return jsonify(create_api_response(
                success=False,
                message="Use POST with JSON body or DELETE with query parameter"
            )), 400
        
        # Use tracker's remove method if available
        if hasattr(tracker, 'remove_reference_image'):
            result = tracker.remove_reference_image(image_name)
            
            if result.get('success', False):
                return jsonify(create_api_response(
                    success=True,
                    message=f"Reference image removed successfully",
                    data={
                        "removed_key": result.get('removed_key'),
                        "remaining_count": result.get('remaining_references', 0)
                    }
                ))
            else:
                return jsonify(create_api_response(
                    success=False,
                    message=f"Failed to remove reference: {result.get('error', 'Unknown error')}"
                )), 404
        else:
            # Fallback: manual reference management
            if not hasattr(tracker, 'reference_data'):
                return jsonify(create_api_response(
                    success=False,
                    message="No reference storage available"
                )), 503
            
            # Determine which reference to remove
            if image_name is None:
                if hasattr(tracker, 'default_reference_key') and tracker.default_reference_key:
                    key_to_remove = tracker.default_reference_key
                else:
                    return jsonify(create_api_response(
                        success=False,
                        message="No default reference image to remove"
                    )), 404
            else:
                key_to_remove = image_name
            
            # Check if reference exists
            if key_to_remove not in tracker.reference_data:
                return jsonify(create_api_response(
                    success=False,
                    message=f"Reference image '{key_to_remove}' not found"
                )), 404
            
            # Remove reference
            del tracker.reference_data[key_to_remove]
            
            # Update default reference if necessary
            if hasattr(tracker, 'default_reference_key') and key_to_remove == tracker.default_reference_key:
                if tracker.reference_data:
                    tracker.default_reference_key = next(iter(tracker.reference_data.keys()))
                else:
                    tracker.default_reference_key = None
            
            return jsonify(create_api_response(
                success=True,
                message=f"Reference image removed successfully",
                data={
                    "removed_key": key_to_remove,
                    "remaining_count": len(tracker.reference_data)
                }
            ))
        
    except Exception as e:
        return jsonify(create_api_response(
            success=False,
            message=f"Error removing reference: {str(e)}"
        )), 500

@app.route("/docs")
def api_docs():
    """Simple API documentation."""
    docs_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FlowFormer++ Keypoint Tracking API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #0066cc; }
            .method.post { color: #ff6600; }
            .method.delete { color: #cc0066; }
            .tracker-compatible { border-left: 4px solid #00cc66; }
        </style>
    </head>
    <body>
        <h1>FlowFormer++ Keypoint Tracking API</h1>
        
        <h2>Service Endpoints</h2>
        <div class="endpoint">
            <div class="method">GET /</div>
            <p>Service information page</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET /health</div>
            <p>Service health check and tracker status</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET /references</div>
            <p>List all stored reference images</p>
        </div>
        
        <h2>Tracking Endpoints</h2>
        <div class="endpoint tracker-compatible">
            <div class="method post">POST /set_reference_image</div>
            <p>Set reference image with keypoints</p>
            <p><strong>JSON Body:</strong> {"image_base64": "base64_jpg_string", "keypoints": [...], "image_name": "optional"}</p>
        </div>
        
        <div class="endpoint tracker-compatible">
            <div class="method post">POST /track_keypoints</div>
            <p>Track keypoints using FlowFormer++ model</p>
            <p><strong>JSON Body:</strong> {"image_base64": "base64_jpg_string", "reference_name": "optional", "bidirectional": false}</p>
        </div>
        
        <div class="endpoint tracker-compatible">
            <div class="method post">POST /remove_reference_image</div>
            <div class="method delete">DELETE /remove_reference_image</div>
            <p>Remove reference image by name</p>
            <p><strong>POST JSON:</strong> {"image_name": "optional"} or <strong>DELETE Query:</strong> ?image_name=name</p>
        </div>
        

        
        <div class="endpoint">
            <div class="method">GET /docs</div>
            <p>This API documentation</p>
        </div>
    </body>
    </html>
    """
    return docs_html

def startup_service():
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
    print("📖 Flask routes: /health, /references, /set_reference, /track_keypoints")
    
    startup_service()
    
    app.run(
        host="0.0.0.0",
        port=8001,
        debug=False
    )