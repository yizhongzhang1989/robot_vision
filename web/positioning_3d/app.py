#!/usr/bin/env python3
"""
3D Positioning Service
======================

Multi-view triangulation service with real-time keypoint tracking
using FlowFormer++ and multi-robot support.

Author: Yizhong Zhang
Date: November 2025
"""

import os
import sys
import json
import time
import logging
import threading
import queue as queue_module
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import yaml
import numpy as np
import cv2
import argparse

from flask import Flask, request, jsonify, render_template, Response
import base64
from PIL import Image
import io

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import positioning service modules (using absolute imports from web.positioning_3d)
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from models import (
    RobotSession, View, TrackingTask, ReferenceImage,
    SessionStatus, ViewStatus, CameraParams, TriangulationResult
)
from ffpp_client import FFPPClient
from task_queue import TaskQueueManager
from session_manager import SessionManager

# Import triangulation core
from core.triangulation import triangulate_multiview, triangulate_view_plane

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
ffpp_client: Optional[FFPPClient] = None
task_queue: Optional[TaskQueueManager] = None
session_manager: Optional[SessionManager] = None
config: Optional[Dict] = None
reference_images: Dict[str, ReferenceImage] = {}

# SSE clients for real-time updates
sse_clients = set()

class SSEClient:
    """Server-Sent Events client."""
    def __init__(self):
        self.queue = queue_module.Queue()
        self.active = True
    
    def put(self, data):
        if self.active:
            try:
                self.queue.put_nowait(data)
            except queue_module.Full:
                pass
    
    def close(self):
        self.active = False


def load_config() -> Dict:
    """Load service configuration from YAML file."""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def broadcast_sse_event(event_type: str, data: dict):
    """Broadcast an event to all connected SSE clients."""
    global sse_clients
    
    event_data = {
        'type': event_type,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    
    # Remove disconnected clients
    disconnected_clients = set()
    for client in sse_clients:
        if not client.active:
            disconnected_clients.add(client)
        else:
            client.put(event_data)
    
    sse_clients -= disconnected_clients


def parse_dataset_references(dataset_path: str) -> List[ReferenceImage]:
    """
    Parse dataset directory to find reference images and keypoints.
    
    Dataset structure:
        dataset_path/
            ref_name_1/
                ref_img_1.jpg
                ref_img_1.json
            ref_name_2/
                ref_img_1.jpg
                ref_img_1.json
    
    Args:
        dataset_path: Path to dataset root
        
    Returns:
        List of ReferenceImage objects
    """
    references = []
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return references
    
    logger.info(f"Parsing dataset: {dataset_path}")
    
    # Iterate through subdirectories (each subdir is a reference name)
    for subdir in dataset_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        ref_name = subdir.name
        image_path = subdir / "ref_img_1.jpg"
        json_path = subdir / "ref_img_1.json"
        
        # Check if both files exist
        if not image_path.exists():
            logger.warning(f"Skipping {ref_name}: ref_img_1.jpg not found")
            continue
        
        if not json_path.exists():
            logger.warning(f"Skipping {ref_name}: ref_img_1.json not found")
            continue
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Skipping {ref_name}: Failed to load image")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load JSON with keypoints
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract keypoints from JSON
            keypoints = []
            if 'keypoints' in data:
                # Format: [{'x': float, 'y': float}, ...]
                keypoints = data['keypoints']
            elif 'points' in data:
                # Alternative format
                for pt in data['points']:
                    if isinstance(pt, dict) and 'x' in pt and 'y' in pt:
                        keypoints.append({'x': float(pt['x']), 'y': float(pt['y'])})
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        keypoints.append({'x': float(pt[0]), 'y': float(pt[1])})
            
            if not keypoints:
                logger.warning(f"Skipping {ref_name}: No keypoints found in JSON")
                continue
            
            # Create reference object
            reference = ReferenceImage(
                name=ref_name,
                image_path=str(image_path),
                keypoints=keypoints,
                image=image,
                metadata={
                    'json_path': str(json_path),
                    'num_keypoints': len(keypoints),
                    'image_shape': image.shape
                }
            )
            
            references.append(reference)
            logger.info(f"✅ Loaded reference: {ref_name} with {len(keypoints)} keypoints")
            
        except Exception as e:
            logger.warning(f"Error parsing {ref_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    logger.info(f"Found {len(references)} reference images in dataset")
    return references


def upload_references_to_ffpp(references: List[ReferenceImage]) -> bool:
    """
    Upload all reference images to FFPP server.
    
    Args:
        references: List of ReferenceImage objects
        
    Returns:
        True if all uploads successful
    """
    global ffpp_client, reference_images
    
    if not ffpp_client:
        logger.error("FFPP client not initialized")
        return False
    
    logger.info(f"Uploading {len(references)} references to FFPP server...")
    
    all_success = True
    for ref in references:
        logger.info(f"Uploading reference: {ref.name}")
        
        result = ffpp_client.set_reference_image(
            image=ref.image,
            keypoints=ref.keypoints,
            image_name=ref.name
        )
        
        if result.get('success'):
            ref.uploaded_to_ffpp = True
            reference_images[ref.name] = ref
            logger.info(f"✅ Uploaded: {ref.name}")
        else:
            all_success = False
            logger.error(f"❌ Failed to upload {ref.name}: {result.get('error')}")
    
    logger.info(f"Upload complete: {len(reference_images)}/{len(references)} successful")
    return all_success


def handle_tracking_task(task: TrackingTask) -> bool:
    """
    Handle a tracking task from the queue.
    
    Args:
        task: TrackingTask to process
        
    Returns:
        True if tracking successful
    """
    global ffpp_client, session_manager
    
    logger.info(f"Handling tracking task: {task.task_id}")
    
    # Update view status to tracking
    session_manager.update_view_status(
        task.session_id,
        task.view_id,
        ViewStatus.TRACKING
    )
    
    # Broadcast status update
    session = session_manager.get_session(task.session_id)
    if session:
        broadcast_sse_event('view_tracking', {
            'session_id': task.session_id,
            'view_id': task.view_id,
            'status': 'tracking'
        })
    
    # Call FFPP to track keypoints
    result = ffpp_client.track_keypoints(
        image_base64=task.image_base64,
        reference_name=task.reference_name
    )
    
    if result.get('success'):
        # Extract tracked keypoints from FFPP result
        result_data = result.get('result', {})
        tracked_keypoints = result_data.get('tracked_keypoints', [])
        
        # Convert to standard format with embedded accuracy
        keypoints_2d = []
        
        for kp in tracked_keypoints:
            if isinstance(kp, list) and len(kp) >= 2:
                # Old format: [x, y]
                keypoints_2d.append({'x': float(kp[0]), 'y': float(kp[1])})
            elif isinstance(kp, dict):
                # New format: {'x': x, 'y': y, 'consistency_distance': dist, ...}
                if 'x' in kp and 'y' in kp:
                    keypoint_dict = {'x': float(kp['x']), 'y': float(kp['y'])}
                    
                    # Embed consistency_distance (smaller is better)
                    consistency_dist = kp.get('consistency_distance')
                    if consistency_dist is not None:
                        keypoint_dict['consistency_distance'] = float(consistency_dist)
                    
                    keypoints_2d.append(keypoint_dict)
        
        # Update view with keypoints (accuracy embedded in each point)
        session_manager.update_view_status(
            task.session_id,
            task.view_id,
            ViewStatus.TRACKED,
            keypoints_2d=keypoints_2d
        )
        
        logger.info(f"✅ Tracked {len(keypoints_2d)} keypoints for view {task.view_id}")
        
        # Broadcast update
        broadcast_sse_event('view_tracked', {
            'session_id': task.session_id,
            'view_id': task.view_id,
            'keypoints_count': len(keypoints_2d)
        })
        
        return True
    else:
        # Tracking failed
        error_msg = result.get('error', 'Unknown error')
        logger.error(f"❌ Tracking failed for view {task.view_id}: {error_msg}")
        
        session_manager.update_view_status(
            task.session_id,
            task.view_id,
            ViewStatus.FAILED,
            error_message=error_msg
        )
        
        # Broadcast failure
        broadcast_sse_event('view_failed', {
            'session_id': task.session_id,
            'view_id': task.view_id,
            'error': error_msg
        })
        
        return False


def trigger_triangulation(session_id: str, wait_for_tracking: bool = True, timeout_ms: int = 0):
    """
    Trigger triangulation for a session on-demand.
    
    This function is called when get_result() is invoked. It will wait for
    pending views to be tracked (if wait_for_tracking=True), then perform
    triangulation with all currently tracked views.
    
    Args:
        session_id: Session identifier
        wait_for_tracking: If True, wait for pending views to be tracked
        timeout_ms: Maximum wait time in milliseconds for tracking to complete
    
    Returns:
        True if triangulation started/completed, False if insufficient views or timeout
    """
    global session_manager
    
    session = session_manager.get_session(session_id)
    if not session:
        return False
    
    # If wait_for_tracking, wait for pending views to complete
    if wait_for_tracking and timeout_ms > 0:
        start_time = time.time() * 1000
        check_interval = 100  # ms
        
        while True:
            pending_views = [v for v in session.views if v.status in [ViewStatus.RECEIVED, ViewStatus.QUEUED, ViewStatus.TRACKING]]
            
            if len(pending_views) == 0:
                break
            
            elapsed = (time.time() * 1000) - start_time
            if elapsed >= timeout_ms:
                logger.warning(f"Session {session_id}: Timeout waiting for {len(pending_views)} views to be tracked")
                return False
            
            time.sleep(check_interval / 1000.0)
    
    # Count tracked views
    tracked_views = [v for v in session.views if v.status == ViewStatus.TRACKED]
    
    # Need at least 2 tracked views for triangulation
    if len(tracked_views) < 2:
        error_msg = f"Insufficient tracked views for triangulation: {len(tracked_views)}/2 minimum"
        logger.error(f"Session {session_id}: {error_msg}")
        return False
    
    logger.info(f"Session {session_id}: Starting triangulation with {len(tracked_views)} views")
    
    # Update status
    session_manager.update_session_status(session_id, SessionStatus.TRIANGULATING)
    
    # Broadcast update
    broadcast_sse_event('triangulation_started', {
        'session_id': session_id,
        'num_views': len(tracked_views)
    })
    
    # Perform triangulation synchronously (in the calling thread)
    _perform_triangulation(session_id)
    
    return True


def _perform_triangulation(session_id: str):
    """
    Perform triangulation for a session.
    
    Args:
        session_id: Session identifier
    """
    global session_manager
    
    session = session_manager.get_session(session_id)
    if not session:
        return
    
    try:
        start_time = time.time()
        
        # Prepare view data for triangulation
        view_data = []
        for view in session.views:
            if view.status == ViewStatus.TRACKED and view.keypoints_2d:
                # Convert keypoints to array format
                points_2d = np.array([[kp['x'], kp['y']] for kp in view.keypoints_2d])
                
                view_dict = {
                    'points_2d': points_2d,
                    'image_size': view.camera_params.image_size,
                    'intrinsic': view.camera_params.intrinsic,
                    'extrinsic': view.camera_params.extrinsic
                }
                
                # Add distortion if available
                if view.camera_params.distortion is not None:
                    view_dict['distortion'] = view.camera_params.distortion
                
                view_data.append(view_dict)
        
        if len(view_data) < 2:
            raise ValueError("Need at least 2 tracked views for triangulation")
        
        logger.info(f"Triangulating with {len(view_data)} views...")
        
        # Call triangulation
        result = triangulate_multiview(view_data)
        
        processing_time = time.time() - start_time
        
        if result.get('success'):
            # Extract results
            points_3d = result['points_3d']
            reprojection_errors = result['reprojection_errors']
            
            # Calculate mean error
            mean_error = np.mean([np.mean(errors) for errors in reprojection_errors])
            
            # Create result object
            triangulation_result = TriangulationResult(
                success=True,
                points_3d=points_3d,
                reprojection_errors=reprojection_errors,
                mean_error=float(mean_error),
                processing_time=processing_time
            )
            
            # Update session
            session.result = triangulation_result
            session_manager.update_session_status(session_id, SessionStatus.COMPLETED)
            
            logger.info(f"✅ Triangulation completed: {len(points_3d)} points, {mean_error:.3f}px error")
            
            # Broadcast completion
            broadcast_sse_event('triangulation_completed', {
                'session_id': session_id,
                'num_points': len(points_3d),
                'mean_error': float(mean_error),
                'processing_time': processing_time
            })
        else:
            error_msg = result.get('error_message', 'Unknown error')
            
            triangulation_result = TriangulationResult(
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
            
            session.result = triangulation_result
            session_manager.update_session_status(session_id, SessionStatus.FAILED, error_msg)
            
            logger.error(f"❌ Triangulation failed: {error_msg}")
            
            broadcast_sse_event('triangulation_failed', {
                'session_id': session_id,
                'error': error_msg
            })
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Triangulation exception: {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        
        triangulation_result = TriangulationResult(
            success=False,
            error_message=error_msg
        )
        
        session.result = triangulation_result
        session_manager.update_session_status(session_id, SessionStatus.FAILED, error_msg)
        
        broadcast_sse_event('triangulation_failed', {
            'session_id': session_id,
            'error': error_msg
        })


# Initialize Flask app
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')


@app.route("/")
def dashboard():
    """Main dashboard."""
    service_status = get_service_status()
    return render_template('dashboard.html', 
                         service_status=service_status)


@app.route("/health")
def health_check():
    """Health check endpoint."""
    status = get_service_status()
    return jsonify({
        'success': True,
        'service': '3D Positioning Service',
        'status': status,
        'timestamp': datetime.now().isoformat()
    })


def get_service_status() -> Dict:
    """Get comprehensive service status."""
    global ffpp_client, task_queue, session_manager, reference_images
    
    status = {
        'service': '3D Positioning Service',
        'version': '1.0.0',
        'port': config.get('service', {}).get('port', 8004),
        'timestamp': datetime.now().isoformat()
    }
    
    # FFPP status
    if ffpp_client:
        ffpp_healthy = ffpp_client.health_check()
        status['ffpp_server'] = {
            'connected': ffpp_healthy,
            'host': ffpp_client.host,
            'port': ffpp_client.port
        }
    else:
        status['ffpp_server'] = {'connected': False}
    
    # Queue status
    if task_queue:
        status['queue'] = task_queue.get_statistics()
    
    # Session status
    if session_manager:
        status['sessions'] = session_manager.get_statistics()
    
    # References
    status['references'] = {
        'loaded': len(reference_images),
        'names': list(reference_images.keys())
    }
    
    return status


@app.route("/init_session", methods=["POST"])
def init_session():
    """
    Initialize a new positioning session.
    
    Request body:
    {
        "reference_name": "checkerboard_11x8"
    }
    """
    global session_manager, reference_images
    
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'reference_name' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: reference_name'
            }), 400
        
        reference_name = data['reference_name']
        
        # Validate reference exists
        if reference_name not in reference_images:
            return jsonify({
                'success': False,
                'error': f'Reference "{reference_name}" not found',
                'available_references': list(reference_images.keys())
            }), 404
        
        # Create session
        session = session_manager.create_session(
            reference_name=reference_name
        )
        
        logger.info(f"Created session {session.session_id}")
        
        # Broadcast event
        broadcast_sse_event('session_created', {
            'session_id': session.session_id
        })
        
        return jsonify({
            'success': True,
            'session_id': session.session_id,
            'status': session.status.value,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/upload_view", methods=["POST"])
def upload_view():
    """
    Upload a single view for triangulation.
    
    Request body:
    {
        "session_id": "robot_arm_01_1732012345_abc123",
        "view_id": "view_0",
        "image_base64": "data:image/png;base64,...",
        "camera_params": {
            "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "extrinsic": [[...], ...],
            "distortion": [k1, k2, p1, p2, k3],  // optional
            "image_size": [width, height]
        }
    }
    """
    global session_manager, task_queue
    
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['session_id', 'view_id', 'image_base64', 'camera_params']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        session_id = data['session_id']
        view_id = data['view_id']
        image_base64 = data['image_base64']
        camera_params_dict = data['camera_params']
        
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({
                'success': False,
                'error': f'Session not found: {session_id}'
            }), 404
        
        # Parse camera parameters
        camera_params = CameraParams(
            intrinsic=np.array(camera_params_dict['intrinsic']),
            extrinsic=np.array(camera_params_dict['extrinsic']),
            distortion=np.array(camera_params_dict['distortion']) if 'distortion' in camera_params_dict else None,
            image_size=tuple(camera_params_dict.get('image_size', (0, 0)))
        )
        
        # Add view to session
        view = session_manager.add_view(
            session_id=session_id,
            view_id=view_id,
            image_base64=image_base64,
            camera_params=camera_params
        )
        
        if not view:
            return jsonify({
                'success': False,
                'error': 'Failed to add view to session'
            }), 500
        
        # Create tracking task
        task_id = f"{session_id}_{view_id}_{int(time.time())}"
        task = TrackingTask(
            task_id=task_id,
            session_id=session_id,
            view_id=view_id,
            reference_name=session.reference_name,
            image_base64=image_base64
        )
        
        # Enqueue task
        enqueued = task_queue.enqueue(task)
        
        if enqueued:
            # Update view status
            session_manager.update_view_status(
                session_id,
                view_id,
                ViewStatus.QUEUED,
            )
            
            queue_position = task_queue.get_queue_size()
            view.queue_position = queue_position
            
            logger.info(f"View {view_id} enqueued (position: {queue_position})")
            
            # Broadcast event
            broadcast_sse_event('view_uploaded', {
                'session_id': session_id,
                'view_id': view_id,
                'queue_position': queue_position
            })
            
            return jsonify({
                'success': True,
                'view_id': view_id,
                'status': ViewStatus.QUEUED.value,
                'queue_position': queue_position,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Queue full, please try again later'
            }), 503
            
    except Exception as e:
        logger.error(f"Error uploading view: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/session_status/<session_id>")
def session_status(session_id: str):
    """Get status of a specific session."""
    global session_manager
    
    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({
            'success': False,
            'error': f'Session not found: {session_id}'
        }), 404
    
    return jsonify({
        'success': True,
        'session': session.to_dict(include_views=True, include_images=False),
        'timestamp': datetime.now().isoformat()
    })


@app.route("/result/<session_id>")
def get_result(session_id: str):
    """Get triangulation result for a session. Triggers triangulation on-demand."""
    global session_manager
    
    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({
            'success': False,
            'error': f'Session not found: {session_id}'
        }), 404
    
    # Get timeout from query parameter (default: 30 seconds)
    timeout_ms = int(request.args.get('timeout', 30000))
    
    # Trigger triangulation if not already completed
    if session.status != SessionStatus.COMPLETED:
        success = trigger_triangulation(session_id, wait_for_tracking=True, timeout_ms=timeout_ms)
        
        if not success:
            # Check why it failed
            tracked_views = [v for v in session.views if v.status == ViewStatus.TRACKED]
            pending_views = [v for v in session.views if v.status in [ViewStatus.RECEIVED, ViewStatus.QUEUED, ViewStatus.TRACKING]]
            
            if len(tracked_views) < 2:
                return jsonify({
                    'success': False,
                    'error': f'Insufficient tracked views for triangulation (have {len(tracked_views)}, need at least 2)',
                    'session': session.to_dict(include_views=True, include_images=False)
                }), 400
            elif len(pending_views) > 0:
                return jsonify({
                    'success': False,
                    'error': f'Timeout waiting for {len(pending_views)} views to be tracked',
                    'session': session.to_dict(include_views=True, include_images=False)
                }), 408
            else:
                return jsonify({
                    'success': False,
                    'error': 'Triangulation failed',
                    'session': session.to_dict(include_views=True, include_images=False)
                }), 500
    
    # Check if triangulation completed successfully
    if session.status != SessionStatus.COMPLETED:
        return jsonify({
            'success': False,
            'error': f'Triangulation did not complete (status: {session.status.value})',
            'session': session.to_dict(include_views=True, include_images=False)
        }), 500
    
    # Collect per-view 2D keypoints from tracked views (in upload order)
    views_data = []
    for view in session.views:
        if view.status == ViewStatus.TRACKED and view.keypoints_2d:
            views_data.append({
                'keypoints_2d': view.keypoints_2d
            })
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'result': session.result.to_dict() if session.result else None,
        'views': views_data,
        'timestamp': datetime.now().isoformat()
    })


@app.route("/result_view_plane", methods=["POST"])
def get_result_view_plane():
    """
    Get view-plane triangulation result for a single-view session.
    
    Projects 2D points from a single camera view onto a known 3D plane.
    """
    global session_manager
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        plane_point = data.get('plane_point')
        plane_normal = data.get('plane_normal')
        
        # Validate input
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Missing session_id'
            }), 400
        
        if plane_point is None or plane_normal is None:
            return jsonify({
                'success': False,
                'error': 'Missing plane_point or plane_normal'
            }), 400
        
        # Convert to numpy arrays
        plane_point = np.array(plane_point, dtype=np.float64)
        plane_normal = np.array(plane_normal, dtype=np.float64)
        
        # Validate shapes
        if plane_point.shape != (3,):
            return jsonify({
                'success': False,
                'error': f'plane_point must have shape (3,), got {plane_point.shape}'
            }), 400
        
        if plane_normal.shape != (3,):
            return jsonify({
                'success': False,
                'error': f'plane_normal must have shape (3,), got {plane_normal.shape}'
            }), 400
        
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({
                'success': False,
                'error': f'Session not found: {session_id}'
            }), 404
        
        # Check that exactly 1 view is uploaded
        tracked_views = [v for v in session.views if v.status == ViewStatus.TRACKED and v.keypoints_2d]
        
        if len(tracked_views) == 0:
            return jsonify({
                'success': False,
                'error': 'No tracked views available. Wait for tracking to complete.'
            }), 400
        
        if len(tracked_views) > 1:
            return jsonify({
                'success': False,
                'error': f'View-plane triangulation requires exactly 1 view, but {len(tracked_views)} views were tracked'
            }), 400
        
        # Get the single view
        view = tracked_views[0]
        
        # Prepare view data for triangulation
        points_2d = np.array([[kp['x'], kp['y']] for kp in view.keypoints_2d])
        
        view_dict = {
            'points_2d': points_2d,
            'image_size': view.camera_params.image_size,
            'intrinsic': view.camera_params.intrinsic,
            'extrinsic': view.camera_params.extrinsic
        }
        
        # Add distortion if available
        if view.camera_params.distortion is not None:
            view_dict['distortion'] = view.camera_params.distortion
        
        logger.info(f"View-plane triangulation: {len(points_2d)} points onto plane")
        
        # Call view-plane triangulation
        start_time = time.time()
        result = triangulate_view_plane(
            view_data=view_dict,
            plane_point=plane_point,
            plane_normal=plane_normal
        )
        processing_time = time.time() - start_time
        
        if not result.get('success'):
            return jsonify({
                'success': False,
                'error': result.get('error_message', 'View-plane triangulation failed')
            }), 500
        
        # Extract results
        points_3d = result['points_3d']
        distances = result['distances']
        
        # Collect per-view 2D keypoints
        views_data = [{
            'keypoints_2d': view.keypoints_2d
        }]
        
        logger.info(f"✅ View-plane triangulation completed: {len(points_3d)} points")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'result': {
                'points_3d': points_3d.tolist(),
                'distances': distances.tolist(),
                'processing_time': processing_time,
                'plane_point': plane_point.tolist(),
                'plane_normal': plane_normal.tolist()
            },
            'views': views_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in view-plane triangulation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/queue_status")
def queue_status():
    """Get current queue status."""
    global task_queue
    
    return jsonify({
        'success': True,
        'queue': task_queue.get_statistics(),
        'timestamp': datetime.now().isoformat()
    })


@app.route("/list_sessions")
def list_sessions():
    """List all sessions with optional filters."""
    global session_manager
    
    robot_id = request.args.get('robot_id')
    status = request.args.get('status')
    
    # Convert status string to enum
    status_filter = None
    if status:
        try:
            status_filter = SessionStatus(status)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid status: {status}'
            }), 400
    
    sessions = session_manager.list_sessions(robot_id=robot_id, status=status_filter)
    
    return jsonify({
        'success': True,
        'sessions': [s.to_dict(include_views=False) for s in sessions],
        'count': len(sessions),
        'timestamp': datetime.now().isoformat()
    })


@app.route("/terminate_session/<session_id>", methods=["POST"])
def terminate_session(session_id: str):
    """Terminate a session and remove its data."""
    global session_manager
    
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({
                'success': False,
                'error': f'Session not found: {session_id}'
            }), 404
        
        # Remove session
        removed = session_manager.remove_session(session_id)
        
        if removed:
            logger.info(f"Terminated session {session_id}")
            
            # Broadcast event
            broadcast_sse_event('session_terminated', {
                'session_id': session_id
            })
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Session terminated and data removed',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to remove session'
            }), 500
            
    except Exception as e:
        logger.error(f"Error terminating session: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/list_references")
def list_references_endpoint():
    """List available reference images."""
    global reference_images
    
    return jsonify({
        'success': True,
        'references': {name: ref.to_dict() for name, ref in reference_images.items()},
        'count': len(reference_images),
        'timestamp': datetime.now().isoformat()
    })


@app.route("/upload_references", methods=["POST"])
def upload_references_endpoint():
    """Manually trigger reference upload to FFPP server."""
    global config, ffpp_client, reference_images
    
    try:
        if not ffpp_client:
            return jsonify({
                'success': False,
                'error': 'FFPP client not initialized'
            }), 500
        
        # Check FFPP connection
        if not ffpp_client.health_check():
            return jsonify({
                'success': False,
                'error': 'FFPP server not connected'
            }), 503
        
        # Parse dataset
        dataset_path = config.get('dataset', {}).get('path', 'output/dataset')
        logger.info(f"Parsing dataset for manual upload: {dataset_path}")
        references = parse_dataset_references(dataset_path)
        
        if not references:
            return jsonify({
                'success': False,
                'error': 'No reference images found in dataset'
            }), 404
        
        # Upload to FFPP
        success = upload_references_to_ffpp(references)
        
        return jsonify({
            'success': success,
            'references_loaded': len(reference_images),
            'references_found': len(references),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error uploading references: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/events")
def sse_events():
    """Server-Sent Events endpoint for real-time updates."""
    def event_stream():
        global sse_clients
        client = SSEClient()
        sse_clients.add(client)
        
        try:
            # Send connection message
            yield f"data: {json.dumps({'type': 'connected', 'message': 'Real-time monitoring started'})}\n\n"
            
            # Stream events
            while client.active:
                try:
                    event = client.queue.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue_module.Empty:
                    # Keepalive ping
                    yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': datetime.now().isoformat()})}\n\n"
                    
        except Exception as e:
            logger.error(f"SSE client error: {e}")
        finally:
            client.close()
            sse_clients.discard(client)
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )


def initialize_service(ffpp_url: Optional[str] = None, dataset_path: Optional[str] = None):
    """Initialize the 3D positioning service.
    
    Args:
        ffpp_url: Optional FFPP server URL (e.g., http://hostname:8001)
        dataset_path: Optional dataset path containing reference subdirs
    """
    global config, ffpp_client, task_queue, session_manager
    
    logger.info("=" * 60)
    logger.info("🚀 Starting 3D Positioning Service")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        sys.exit(1)
    
    # Initialize FFPP client
    ffpp_config = config.get('ffpp_server', {})
    
    # Parse FFPP URL if provided via CLI
    if ffpp_url:
        logger.info(f"Using FFPP server from CLI: {ffpp_url}")
        # Parse URL to extract host and port
        from urllib.parse import urlparse
        parsed = urlparse(ffpp_url)
        ffpp_host = parsed.hostname or parsed.netloc.split(':')[0]
        ffpp_port = parsed.port or 8001
    else:
        ffpp_host = ffpp_config.get('host', 'localhost')
        ffpp_port = ffpp_config.get('port', 8001)
    
    ffpp_client = FFPPClient(
        host=ffpp_host,
        port=ffpp_port,
        timeout=ffpp_config.get('timeout', 30)
    )
    
    # Check FFPP server connection
    logger.info("Connecting to FFPP server...")
    retry_attempts = ffpp_config.get('retry_attempts', 3)
    retry_delay = ffpp_config.get('connection_retry_delay', 5)
    require_ffpp = ffpp_config.get('required', True)
    
    ffpp_connected = False
    for attempt in range(retry_attempts):
        try:
            if ffpp_client.health_check():
                logger.info("✅ FFPP server connected")
                ffpp_connected = True
                break
        except KeyboardInterrupt:
            logger.warning("⚠️ Startup interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
        
        if attempt < retry_attempts - 1:
            logger.warning(f"FFPP server not ready, retrying in {retry_delay}s... ({attempt + 1}/{retry_attempts})")
            try:
                time.sleep(retry_delay)
            except KeyboardInterrupt:
                logger.warning("⚠️ Startup interrupted by user")
                sys.exit(0)
        else:
            if require_ffpp:
                logger.error("❌ Failed to connect to FFPP server")
                sys.exit(1)
            else:
                logger.warning("⚠️ FFPP server not available - service will start in degraded mode")
                break
    
    # Parse and upload references (only if FFPP is connected)
    dataset_config = config.get('dataset', {})
    
    # Use CLI argument or config file
    final_dataset_path = dataset_path or dataset_config.get('path', 'output/dataset')
    
    if ffpp_connected and dataset_config.get('auto_upload_on_start', True):
        if final_dataset_path:
            logger.info(f"Parsing dataset: {final_dataset_path}")
            references = parse_dataset_references(final_dataset_path)
            
            if references:
                upload_references_to_ffpp(references)
            else:
                logger.warning("No reference images found in dataset")
        else:
            logger.warning("No dataset path configured")
    elif not ffpp_connected:
        logger.info("Skipping reference upload (FFPP not connected)")
    
    # Store dataset path in config for later use
    config['dataset']['path'] = final_dataset_path
    
    # Initialize task queue
    queue_config = config.get('queue', {})
    task_queue = TaskQueueManager(max_size=queue_config.get('max_size', 100))
    task_queue.start(handle_tracking_task)
    
    # Initialize session manager
    session_config = config.get('session', {})
    session_manager = SessionManager(
        session_timeout_minutes=session_config.get('timeout_minutes', 30),
        max_sessions=session_config.get('max_concurrent_sessions', 50)
    )
    
    # Start cleanup thread
    cleanup_interval = session_config.get('cleanup_interval_seconds', 300)
    def cleanup_loop():
        while True:
            time.sleep(cleanup_interval)
            session_manager.cleanup_sessions()
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True, name="SessionCleanup")
    cleanup_thread.start()
    
    logger.info("Service initialization complete. References loaded: %d", len(reference_images))


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='3D Positioning Service - Multi-view triangulation with FFPP tracking'
    )
    parser.add_argument(
        '--ffpp-url',
        type=str,
        default=None,
        help='FFPP server URL (e.g., http://hostname:8001 or http://192.168.1.100:8001)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Service port (default: 8004 from config)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Service host (default: 0.0.0.0 from config)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Dataset path containing reference image subdirs (default: output/dataset)'
    )
    
    args = parser.parse_args()
    
    # Initialize service with FFPP URL and dataset path
    initialize_service(ffpp_url=args.ffpp_url, dataset_path=args.dataset_path)
    
    # Get service configuration
    service_config = config.get('service', {})
    
    # Override with CLI arguments if provided
    host = args.host or service_config.get('host', '0.0.0.0')
    port = args.port or service_config.get('port', 8004)
    debug = service_config.get('debug', False)
    
    logger.info("=" * 60)
    logger.info("✅ 3D Positioning Service Ready!")
    logger.info(f"🌐 Access at: http://localhost:{port}")
    logger.info(f"📊 Dashboard: http://localhost:{port}/")
    logger.info(f"🔧 References loaded: {len(reference_images)}")
    logger.info("=" * 60)
    logger.info(f"Starting Flask server on {host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=debug
    )

