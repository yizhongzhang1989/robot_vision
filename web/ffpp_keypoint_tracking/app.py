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
import shutil
import uuid
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
from PIL import Image, ImageDraw
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

# API call logging system
api_call_log = deque(maxlen=50)  # Keep last 50 API calls

def log_api_call(endpoint: str, method: str, data: dict, result: dict, processing_time: float):
    """Log API call for dashboard display with full image and metadata storage."""
    call_record = {
        'id': len(api_call_log) + 1,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'endpoint': endpoint,
        'method': method,
        'success': result.get('success', False),
        'processing_time': round(processing_time, 3),
        'message': result.get('message', 'No message'),
        'keypoints_count': 0,
        'ref_image_size': None,
        'target_image_size': None,
        'error': result.get('error'),
        'tracked_points': 0,
        
        # Store image URLs instead of base64 data
        'ref_image_url': None,
        'target_image_url': None,
        'original_keypoints': [],
        'tracked_keypoints': [],
        'detailed_metadata': {},
        'visualization_data': None
    }
    
    # Extract detailed info and store images based on endpoint
    if endpoint == 'set_reference_image':
        if 'keypoints' in data and data['keypoints']:
            call_record['keypoints_count'] = len(data['keypoints'])
            call_record['original_keypoints'] = data['keypoints']
        
        if 'image_base64' in data:
            try:
                # Save image to disk and get URL
                call_id = f"call_{call_record['id']}_{int(time.time())}"
                img_url = save_image_and_get_url(data['image_base64'], call_id, 'ref')
                call_record['ref_image_url'] = img_url
                
                # Estimate image size
                img_bytes = data['image_base64'].split(',')[-1]
                estimated_size = len(base64.b64decode(img_bytes)) // 1024  # KB
                call_record['ref_image_size'] = f"{estimated_size}KB"
                
                # Store metadata
                call_record['detailed_metadata'] = {
                    'image_name': data.get('image_name', 'unnamed'),
                    'image_format': 'png_file',
                    'keypoints_provided': len(data.get('keypoints', [])) > 0,
                    'image_url': img_url
                }
            except Exception as e:
                logger.error(f"Failed to save reference image: {str(e)}")
                call_record['ref_image_size'] = "Unknown"
                
    elif endpoint == 'track_keypoints' or endpoint == 'demo_track_keypoints':
        # Store tracking results
        if result.get('success') and 'result' in result:
            tracked_kps = result['result'].get('tracked_keypoints', [])
            call_record['tracked_points'] = len(tracked_kps) if tracked_kps else 0
            call_record['tracked_keypoints'] = tracked_kps
            
            # Store detailed metadata
            call_record['detailed_metadata'] = {
                'flow_magnitude': result['result'].get('flow_magnitude', 0),
                'processing_time_detailed': result['result'].get('processing_time', 0),
                'validation_enabled': data.get('enable_validation', False),
                'high_accuracy_mode': data.get('high_accuracy', False)
            }
        
        # Handle both web format and API format for images
        call_id = f"call_{call_record['id']}_{int(time.time())}"
        
        # Save reference image if present
        if 'ref_image' in data:
            try:
                ref_url = save_image_and_get_url(data['ref_image'], call_id, 'ref')
                call_record['ref_image_url'] = ref_url
                ref_size = len(base64.b64decode(data['ref_image'].split(',')[-1])) // 1024
                call_record['ref_image_size'] = f"{ref_size}KB"
                call_record['keypoints_count'] = len(data.get('keypoints', []))
                call_record['original_keypoints'] = data.get('keypoints', [])
            except Exception as e:
                logger.error(f"Failed to save reference image: {str(e)}")
        
        # Save target image if present (either comp_image for web format or image_base64 for API format)
        target_image_key = None
        if 'comp_image' in data:
            target_image_key = 'comp_image'
        elif 'image_base64' in data:
            target_image_key = 'image_base64'
            
        if target_image_key:
            try:
                target_url = save_image_and_get_url(data[target_image_key], call_id, 'target')
                call_record['target_image_url'] = target_url
                target_size = len(base64.b64decode(data[target_image_key].split(',')[-1])) // 1024
                call_record['target_image_size'] = f"{target_size}KB"
            except Exception as e:
                logger.error(f"Failed to save target image: {str(e)}")
    
    api_call_log.appendleft(call_record)  # Add to front (newest first)

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

# Initialize Flask app with template and static folders
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

# Configuration for image storage
IMAGES_DIR = os.path.join(project_root, 'output', 'api_images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Helper function to save image and return URL
def save_image_and_get_url(image_data, call_id, image_type):
    """Save image to disk and return URL path"""
    try:
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Handle base64 data URL
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        elif isinstance(image_data, (bytes, bytearray)):
            image_bytes = image_data
        else:
            # Assume it's already a PIL Image or numpy array
            if hasattr(image_data, 'save'):
                # PIL Image
                buffer = io.BytesIO()
                image_data.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
            else:
                # Numpy array
                pil_image = Image.fromarray(image_data)
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
        
        # Generate filename
        filename = f"{call_id}_{image_type}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # Save to disk
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Return URL path
        return f"/api_images/{filename}"
    
    except Exception as e:
        logger.error(f"Failed to save image {call_id}_{image_type}: {str(e)}")
        return None

# Route to serve API images
@app.route('/api_images/<filename>')
def serve_api_image(filename):
    """Serve API images from the images directory"""
    return send_from_directory(IMAGES_DIR, filename)

# Helper function to create API responses
def create_api_response(success: bool, message: str, result: Optional[Dict] = None, error: Optional[str] = None) -> Dict:
    return {
        'success': success,
        'message': message,
        'result': result,
        'error': error,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode base64 encoded PNG/JPG image and convert to numpy array."""
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

def encode_numpy_array_to_base64(array: np.ndarray) -> Dict:
    """Encode numpy array to base64 string with metadata for exact reconstruction.
    
    Args:
        array: NumPy array to encode
        
    Returns:
        Dict with 'data' (base64 string), 'shape', 'dtype' for reconstruction
    """
    try:
        # Convert array to bytes
        array_bytes = array.tobytes()
        
        # Encode to base64
        array_base64 = base64.b64encode(array_bytes).decode('utf-8')
        
        return {
            'data': array_base64,
            'shape': array.shape,
            'dtype': str(array.dtype)
        }
    except Exception as e:
        raise ValueError(f"Failed to encode numpy array to base64: {str(e)}")

def get_service_status() -> Dict:
    """Get comprehensive service status information."""
    # Try to initialize tracker if not already done
    if not tracker_initialized:
        initialize_tracker()
    
    # Basic status
    status = {
        "service": "FlowFormer++ Keypoint Tracking Service",
        "status": "ready" if tracker_initialized else "error",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "version": "2.0.0",
        "port": 8001,
        "tracker_loaded": tracker_initialized,
        "gpu_available": False,
        "device": "unknown"
    }
    
    # Enhanced status if tracker is available
    if tracker_initialized and tracker:
        status.update({
            "gpu_available": tracker.device.type == 'cuda' if hasattr(tracker, 'device') and tracker.device else False,
            "device": str(tracker.device) if hasattr(tracker, 'device') and tracker.device else "cpu"
        })
        
        # Model information
        status["model_info"] = {
            "Model": "FlowFormer++",
            "Status": "Loaded" if tracker_initialized else "Not Available",
            "Device": status["device"],
            "Max Image Size": getattr(tracker, 'max_image_size', 'Unknown')
        }
        
        # System information
        status["system_info"] = {
            "References Stored": len(tracker.reference_data) if hasattr(tracker, 'reference_data') else 0,
            "Default Reference": getattr(tracker, 'default_reference_key', None) or "None",
            "Service Uptime": "Active",
            "Memory Usage": "Monitoring Active"
        }
    else:
        status["error"] = "Tracker initialization failed"
        status["model_info"] = {
            "Model": "FlowFormer++",
            "Status": "Failed to Load",
            "Device": "Unknown",
            "Error": "Import or initialization error"
        }
        status["system_info"] = {
            "Status": "Service Degraded",
            "Available": False
        }
    
    return status

@app.route("/")
def dashboard():
    """Main dashboard with API call monitoring interface."""
    # Get service status for dashboard
    status_info = get_service_status()
    
    # Get recent API calls (last 5 for main dashboard)
    recent_calls = list(api_call_log)[:5]
    
    return render_template('dashboard.html', 
                         service_status=status_info,
                         tracker_initialized=tracker_initialized,
                         recent_api_calls=recent_calls,
                         total_api_calls=len(api_call_log))

@app.route("/status")
def status_endpoint():
    """Get service status for AJAX requests."""
    return jsonify(get_service_status())

@app.route("/get_session_data")
def get_session_data():
    """Get any existing session data for the frontend."""
    # This could be expanded to maintain session state
    return jsonify(create_api_response(
        success=True,
        message="No session data available",
        result={"keypoints": []}
    ))

@app.route("/api_logs")
def get_api_logs():
    """Get API call logs for dashboard."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # Paginate the logs
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    logs_page = list(api_call_log)[start_idx:end_idx]
    
    return jsonify({
        'success': True,
        'logs': logs_page,
        'total': len(api_call_log),
        'page': page,
        'per_page': per_page,
        'has_more': end_idx < len(api_call_log)
    })

@app.route("/validate_tracking", methods=["POST"])
def validate_tracking():
    """Validate tracking results with reverse flow."""
    if not tracker_initialized:
        return jsonify(create_api_response(
            success=False,
            message="Tracker not initialized"
        )), 503
    
    # This would implement bidirectional validation
    # For now, return a placeholder response
    return jsonify(create_api_response(
        success=True,
        message="Validation completed",
        result={
            "reverse_flow_keypoints": [],
            "validation_errors": [],
            "average_error": 0.0,
            "validation_visualization_path": "/static/images/validation_placeholder.png"
        }
    ))

@app.route("/export_results")
def export_results():
    """Export tracking results as JSON file."""
    # This would export the current tracking session results
    # For now, return a placeholder
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "service": "FlowFormer++ Keypoint Tracking",
        "results": "No results available"
    }
    
    from flask import Response
    return Response(
        json.dumps(results, indent=2),
        mimetype='application/json',
        headers={"Content-disposition": "attachment; filename=tracking_results.json"}
    )

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
        result=status
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
        result={
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
    start_time = time.time()
    
    if not tracker_initialized:
        result = create_api_response(
            success=False,
            message="Tracker not initialized. Check /health endpoint."
        )
        log_api_call('set_reference_image', 'POST', {}, result, time.time() - start_time)
        return jsonify(result), 503
    
    try:
        # Get JSON data
        if not request.is_json:
            result = create_api_response(
                success=False,
                message="Request must be JSON with base64 encoded image"
            )
            log_api_call('set_reference_image', 'POST', {}, result, time.time() - start_time)
            return jsonify(result), 400
        
        data = request.get_json()
        
        # Check required fields
        if 'image_base64' not in data:
            result = create_api_response(
                success=False,
                message="No image_base64 field provided"
            )
            log_api_call('set_reference_image', 'POST', data, result, time.time() - start_time)
            return jsonify(result), 400
        
        # Get data fields
        image_base64 = data['image_base64']
        keypoints_data = data.get('keypoints')  # Now optional
        image_name = data.get('image_name')
        
        # Decode and load image
        image_np = decode_base64_image(image_base64)
        
        # Validate keypoints format if provided
        if keypoints_data is not None:
            if not isinstance(keypoints_data, list):
                result = create_api_response(
                    success=False,
                    message="Keypoints must be a list when provided"
                )
                log_api_call('set_reference_image', 'POST', data, result, time.time() - start_time)
                return jsonify(result), 400
            
            # Validate keypoints format
            for kp in keypoints_data:
                if not isinstance(kp, dict) or 'x' not in kp or 'y' not in kp:
                    raise ValueError("Each keypoint must be a dict with 'x' and 'y' keys")
        
        # Use tracker to set reference image
        tracker_result = tracker.set_reference_image(
            image=image_np,
            keypoints=keypoints_data,
            image_name=image_name
        )
        
        # Always return the complete tracker result for debugging, regardless of success/failure
        if tracker_result.get('success', False):
            keypoints_count = len(keypoints_data) if keypoints_data is not None else 0
            result = create_api_response(
                success=True,
                message=f"Reference image set successfully" + (f" with {keypoints_count} keypoints" if keypoints_count > 0 else " (no keypoints provided)"),
                result=tracker_result  # Return the complete tracker result
            )
            log_api_call('set_reference_image', 'POST', data, result, time.time() - start_time)
            return jsonify(result)
        else:
            result = create_api_response(
                success=False,
                message=f"Failed to set reference image: {tracker_result.get('error', 'unknown error')}",
                result=tracker_result  # Return the complete tracker result for debugging
            )
            log_api_call('set_reference_image', 'POST', data, result, time.time() - start_time)
            return jsonify(result), 500
        
    except json.JSONDecodeError as e:
        result = create_api_response(
            success=False,
            message=f"Invalid keypoints JSON: {str(e)}"
        )
        log_api_call('set_reference_image', 'POST', data, result, time.time() - start_time)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"Error in set_reference_image: {str(e)}")
        logger.error(traceback.format_exc())
        result = create_api_response(
            success=False,
            message=f"Error: {str(e)}"
        )
        log_api_call('set_reference_image', 'POST', data, result, time.time() - start_time)
        return jsonify(result), 500

@app.route("/track_keypoints", methods=["POST"])
def track_keypoints():
    """Track keypoints using real FlowFormer++ model - supports both API and Web interface."""
    start_time = time.time()
    
    if not tracker_initialized:
        result = create_api_response(
            success=False,
            message="Tracker not initialized. Check /health endpoint."
        )
        log_api_call('track_keypoints', 'POST', {}, result, time.time() - start_time)
        return jsonify(result), 503
    
    try:
        # Get JSON data
        if not request.is_json:
            result = create_api_response(
                success=False,
                message="Request must be JSON with base64 encoded images and keypoints"
            )
            log_api_call('track_keypoints', 'POST', {}, result, time.time() - start_time)
            return jsonify(result), 400
        
        data = request.get_json()
        
        # Web interface format: includes both ref_image and comp_image + keypoints
        if 'ref_image' in data and 'comp_image' in data and 'keypoints' in data:
            return track_keypoints_web_format(data, start_time)
        
        # Legacy API format: requires pre-set reference image
        else:
            return track_keypoints_api_format(data, start_time)
        
    except Exception as e:
        logger.error(f"Error in track_keypoints: {str(e)}")
        logger.error(traceback.format_exc())
        result = create_api_response(
            success=False,
            message=f"Error: {str(e)}"
        )
        log_api_call('track_keypoints', 'POST', data if 'data' in locals() else {}, result, time.time() - start_time)
        return jsonify(result), 500

def track_keypoints_web_format(data, start_time):
    """Handle web interface tracking format with ref_image, comp_image, and keypoints."""
    try:
        # Extract web format data
        ref_image_b64 = data['ref_image']
        comp_image_b64 = data['comp_image']
        keypoints = data['keypoints']
        
        # Tracking options
        enable_validation = data.get('enable_validation', False)
        visualize_paths = data.get('visualize_paths', False)
        high_accuracy = data.get('high_accuracy', False)
        
        # Decode images
        ref_image_np = decode_base64_image(ref_image_b64)
        comp_image_np = decode_base64_image(comp_image_b64)
        
        # Convert keypoints to proper format
        keypoints_list = []
        for kp in keypoints:
            if isinstance(kp, list) and len(kp) >= 2:
                keypoints_list.append({'x': kp[0], 'y': kp[1]})
            elif isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                keypoints_list.append(kp)
        
        # Set reference image with keypoints
        ref_result = tracker.set_reference_image(
            image=ref_image_np,
            keypoints=keypoints_list,
            image_name="web_session_ref"
        )
        
        if not ref_result.get('success', False):
            return jsonify(create_api_response(
                success=False,
                message=f"Failed to set reference image: {ref_result.get('error', 'Unknown error')}",
                result=ref_result
            )), 500
        
        # Track keypoints
        track_result = tracker.track_keypoints(
            target_image=comp_image_np,
            reference_name="web_session_ref",
            bidirectional=enable_validation,
            return_flow=visualize_paths
        )
        
        if track_result.get('success', False):
            # Enhance result for web interface
            web_result = {
                "tracked_keypoints": track_result.get('tracked_keypoints', []),
                "processing_time": track_result.get('processing_time', 0),
                "flow_magnitude": track_result.get('flow_magnitude', 0),
                "visualization_path": "/static/images/tracking_visualization.png",  # Placeholder
                "success": True
            }
            
            result = create_api_response(
                success=True,
                message="Keypoint tracking completed successfully",
                result=web_result
            )
            log_api_call('track_keypoints', 'POST', data, result, time.time() - start_time)
            return jsonify(result)
        else:
            result = create_api_response(
                success=False,
                message=f"Tracking failed: {track_result.get('error', 'unknown error')}",
                result=track_result
            )
            log_api_call('track_keypoints', 'POST', data, result, time.time() - start_time)
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in web format tracking: {str(e)}")
        result = create_api_response(
            success=False,
            message=f"Web tracking error: {str(e)}"
        )
        log_api_call('track_keypoints', 'POST', data, result, time.time() - start_time)
        return jsonify(result), 500

def track_keypoints_api_format(data, start_time):
    """Handle legacy API format with pre-set reference images."""
    # Check if we have any reference images
    if not hasattr(tracker, 'reference_data') or not tracker.reference_data:
        return jsonify(create_api_response(
            success=False,
            message="No reference images available. Use /set_reference_image endpoint first."
        )), 400
    
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
    return_flow = data.get('return_flow', False)
    
    # Decode target image
    target_image_np = decode_base64_image(image_base64)
    
    # Use tracker to track keypoints
    result = tracker.track_keypoints(
        target_image=target_image_np,
        reference_name=reference_name,
        bidirectional=bidirectional,
        return_flow=return_flow
    )
    
    # Encode flow data for JSON transmission if return_flow was requested
    if result.get('success', False) and return_flow and 'flow_data' in result:
        flow_data = result['flow_data']
        encoded_flow_data = {}
        
        # Encode forward flow
        if 'forward_flow' in flow_data and flow_data['forward_flow'] is not None:
            encoded_flow_data['forward_flow'] = encode_numpy_array_to_base64(flow_data['forward_flow'])
        
        # Encode reverse flow if present
        if 'reverse_flow' in flow_data and flow_data['reverse_flow'] is not None:
            encoded_flow_data['reverse_flow'] = encode_numpy_array_to_base64(flow_data['reverse_flow'])
        
        # Copy stats (these are already JSON-serializable)
        if 'forward_flow_stats' in flow_data:
            encoded_flow_data['forward_flow_stats'] = flow_data['forward_flow_stats']
        if 'reverse_flow_stats' in flow_data:
            encoded_flow_data['reverse_flow_stats'] = flow_data['reverse_flow_stats']
        
        # Replace flow_data with encoded version
        result['flow_data'] = encoded_flow_data
    
    # Prepare enhanced logging data with reference image information
    enhanced_data = data.copy()
    
    # Ensure target image is preserved for logging
    if 'image_base64' in data:
        # Make sure target image data is properly formatted for logging
        target_image_data = data['image_base64']
        if not target_image_data.startswith('data:image'):
            enhanced_data['image_base64'] = f"data:image/png;base64,{target_image_data}"
        else:
            enhanced_data['image_base64'] = target_image_data
    
    # Get the reference image data for visual monitoring
    if hasattr(tracker, 'reference_data') and tracker.reference_data:
        # Determine which reference to use
        ref_name = reference_name or tracker.default_reference_key
        if ref_name and ref_name in tracker.reference_data:
            ref_info = tracker.reference_data[ref_name]
            # Add reference image data to logging (ref_info is a ReferenceImageData object)
            if hasattr(ref_info, 'original_image'):
                # Convert numpy array back to base64 for consistent logging
                ref_image_pil = Image.fromarray(ref_info.original_image)
                buffer = io.BytesIO()
                ref_image_pil.save(buffer, format='PNG')
                ref_image_base64 = base64.b64encode(buffer.getvalue()).decode()
                enhanced_data['ref_image'] = f"data:image/png;base64,{ref_image_base64}"
                
            # Add reference keypoints
            if hasattr(ref_info, 'original_keypoints'):
                enhanced_data['keypoints'] = ref_info.original_keypoints
                
    # Always return the complete tracker result for debugging, regardless of success/failure
    if result.get('success', False):
        api_result = create_api_response(
            success=True,
            message=f"Keypoint tracking completed successfully",
            result=result  # Return the complete tracker result
        )
        log_api_call('track_keypoints', 'POST', enhanced_data, api_result, time.time() - start_time)
        return jsonify(api_result)
    else:
        api_result = create_api_response(
            success=False,
            message=f"Tracking failed: {result.get('error', 'unknown error')}",
            result=result  # Return the complete tracker result for debugging
        )
        log_api_call('track_keypoints', 'POST', enhanced_data, api_result, time.time() - start_time)
        return jsonify(api_result), 500

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
            
            # Always return the complete tracker result for debugging, regardless of success/failure
            if result.get('success', False):
                return jsonify(create_api_response(
                    success=True,
                    message=f"Reference image removed successfully",
                    result=result  # Return the complete tracker result
                ))
            else:
                return jsonify(create_api_response(
                    success=False,
                    message=f"Failed to remove reference: {result.get('error', 'Unknown error')}",
                    result=result  # Return the complete tracker result for debugging
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
                result={
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
    """API documentation page."""
    return render_template('api_docs.html', service_status=get_service_status())

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files explicitly."""
    return send_from_directory(app.static_folder, filename)

@app.route("/demo_api_call", methods=["POST"])
def demo_api_call():
    """Demo endpoint to show API call logging in action."""
    start_time = time.time()
    
    # Simulate some processing
    import random
    processing_delay = random.uniform(0.1, 2.0)
    time.sleep(processing_delay)
    
    # Create mock result
    success = random.choice([True, True, True, False])  # 75% success rate
    
    if success:
        result = create_api_response(
            success=True,
            message="Demo tracking completed successfully",
            result={
                "tracked_keypoints": [[120, 180], [250, 200], [350, 320]],
                "processing_time": processing_delay,
                "flow_magnitude": random.uniform(5.0, 25.0),
                "demo_data": True
            }
        )
    else:
        result = create_api_response(
            success=False,
            message="Demo API call failed",
            error="Simulated error for demonstration purposes"
        )
    
    # Create simple demo images (colored rectangles as placeholders)
    def create_demo_image(color, keypoints, width=400, height=300):
        """Create a simple colored rectangle image with keypoints as base64."""
        from PIL import Image, ImageDraw
        import io
        
        # Create image
        img = Image.new('RGB', (width, height), color)
        draw = ImageDraw.Draw(img)
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            # Draw keypoint as a circle
            radius = 5
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill='red' if 'ref' in color else 'blue', 
                        outline='white')
            # Draw keypoint number
            draw.text((x+8, y-8), str(i+1), fill='white')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    # Generate demo keypoints
    demo_keypoints = [[100, 150], [250, 180], [320, 250]]
    tracked_keypoints = [[120, 180], [250, 200], [350, 320]] if success else []
    
    # Mock request data for logging with actual demo images
    mock_data = {
        "ref_image": create_demo_image('lightblue', demo_keypoints),
        "comp_image": create_demo_image('lightgreen', tracked_keypoints if success else []),
        "keypoints": demo_keypoints,
        "enable_validation": False,
        "visualize_paths": True
    }
    
    # Log the API call
    log_api_call('demo_track_keypoints', 'POST', mock_data, result, time.time() - start_time)
    
    return jsonify(result)

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