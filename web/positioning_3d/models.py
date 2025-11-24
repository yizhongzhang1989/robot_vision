"""
Data Models for 3D Positioning Service
======================================

Defines data structures for robot sessions, views, tasks, and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np


class SessionStatus(Enum):
    """Status of a robot positioning session."""
    PENDING = "pending"           # Created, waiting for views
    PROCESSING = "processing"     # Receiving/tracking views
    TRIANGULATING = "triangulating"  # Computing 3D points
    COMPLETED = "completed"       # Successfully completed
    FAILED = "failed"            # Failed with error
    TIMEOUT = "timeout"          # Session timed out


class ViewStatus(Enum):
    """Status of a single camera view."""
    RECEIVED = "received"        # Image received, not yet queued
    QUEUED = "queued"           # In tracking queue
    TRACKING = "tracking"        # Currently being tracked by FFPP
    TRACKED = "tracked"         # Keypoints successfully tracked
    FAILED = "failed"           # Tracking failed


@dataclass
class CameraParams:
    """Camera parameters for a single view."""
    intrinsic: np.ndarray        # 3x3 intrinsic matrix
    extrinsic: np.ndarray        # 4x4 extrinsic matrix (world to camera)
    distortion: Optional[np.ndarray] = None  # Distortion coefficients
    image_size: tuple = (0, 0)   # (width, height)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'intrinsic': self.intrinsic.tolist() if isinstance(self.intrinsic, np.ndarray) else self.intrinsic,
            'extrinsic': self.extrinsic.tolist() if isinstance(self.extrinsic, np.ndarray) else self.extrinsic,
            'distortion': self.distortion.tolist() if self.distortion is not None and isinstance(self.distortion, np.ndarray) else self.distortion,
            'image_size': self.image_size
        }


@dataclass
class View:
    """Single camera view in a positioning session."""
    view_id: str
    session_id: str
    reference_name: str  # Reference image to use for this view
    image: Optional[np.ndarray] = None
    image_base64: Optional[str] = None  # For storage/display
    camera_params: Optional[CameraParams] = None
    keypoints_2d: Optional[List[Dict[str, float]]] = None  # [{'x': x, 'y': y, 'consistency_distance': dist}, ...]
    status: ViewStatus = ViewStatus.RECEIVED
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    queue_position: Optional[int] = None
    
    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            'view_id': self.view_id,
            'session_id': self.session_id,
            'reference_name': self.reference_name,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'error_message': self.error_message,
            'queue_position': self.queue_position,
            'has_keypoints': self.keypoints_2d is not None,
            'keypoints_count': len(self.keypoints_2d) if self.keypoints_2d else 0
        }
        
        if include_image and self.image_base64:
            data['image_base64'] = self.image_base64
            
        if self.keypoints_2d:
            data['keypoints_2d'] = self.keypoints_2d
            
        if self.camera_params:
            data['camera_params'] = self.camera_params.to_dict()
            
        return data


@dataclass
class TriangulationResult:
    """Result of 3D triangulation."""
    success: bool
    points_3d: Optional[Any] = None  # Can be np.ndarray or List[np.ndarray or None]
    reprojection_errors: Optional[Any] = None  # Can be List[np.ndarray] or List[List[float or None]]
    mean_error: Optional[float] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            'success': self.success,
            'error_message': self.error_message,
            'processing_time': self.processing_time
        }
        
        if self.success and self.points_3d is not None:
            # Handle both np.ndarray and List formats (List may contain None values)
            if isinstance(self.points_3d, np.ndarray):
                data['points_3d'] = self.points_3d.tolist()
            else:
                # It's a list, convert numpy arrays to lists, keep None as None
                data['points_3d'] = [
                    pt.tolist() if pt is not None else None
                    for pt in self.points_3d
                ]
            
            data['num_points'] = len(self.points_3d)
            data['mean_error'] = self.mean_error
            
            if self.reprojection_errors:
                # Handle nested list format with potential None values
                data['reprojection_errors'] = []
                for errors in self.reprojection_errors:
                    if isinstance(errors, np.ndarray):
                        data['reprojection_errors'].append(errors.tolist())
                    elif isinstance(errors, list):
                        # List may contain None values
                        data['reprojection_errors'].append(errors)
                    else:
                        data['reprojection_errors'].append(errors)
                
        return data


@dataclass
class RobotSession:
    """A robot positioning session with multiple views."""
    session_id: str
    robot_id: str
    views: List[View] = field(default_factory=list)
    status: SessionStatus = SessionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[TriangulationResult] = None
    error_message: Optional[str] = None
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current session progress."""
        views_received = len(self.views)
        views_tracked = sum(1 for v in self.views if v.status == ViewStatus.TRACKED)
        views_failed = sum(1 for v in self.views if v.status == ViewStatus.FAILED)
        
        return {
            'views_received': views_received,
            'views_tracked': views_tracked,
            'views_failed': views_failed,
            'views_pending': views_received - views_tracked - views_failed,
            'progress_percent': int((views_tracked / views_received * 100)) if views_received > 0 else 0
        }
    
    def is_ready_for_triangulation(self) -> bool:
        """Check if session has enough tracked views for triangulation."""
        tracked_views = [v for v in self.views if v.status == ViewStatus.TRACKED]
        return len(tracked_views) >= 2  # Minimum 2 views for triangulation
    
    def to_dict(self, include_views: bool = True, include_images: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            'session_id': self.session_id,
            'robot_id': self.robot_id,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'progress': self.get_progress()
        }
        
        if include_views:
            data['views'] = [v.to_dict(include_image=include_images) for v in self.views]
            
        if self.result:
            data['result'] = self.result.to_dict()
            
        return data


@dataclass
class TrackingTask:
    """Task for keypoint tracking in FFPP queue."""
    task_id: str
    session_id: str
    view_id: str
    reference_name: str
    image_base64: str
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'task_id': self.task_id,
            'session_id': self.session_id,
            'view_id': self.view_id,
            'reference_name': self.reference_name,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'retry_count': self.retry_count
        }


@dataclass
class ReferenceImage:
    """Reference image configuration."""
    name: str
    image_path: str
    keypoints: List[Dict[str, float]]
    image: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    uploaded_to_ffpp: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'image_path': self.image_path,
            'keypoints_count': len(self.keypoints),
            'metadata': self.metadata,
            'uploaded_to_ffpp': self.uploaded_to_ffpp
        }
