"""
Robot Vision Core Module
=======================

Core computer vision functionality for robotics applications.
"""

from .keypoint_tracker import KeypointTracker
from .ffpp_keypoint_tracker import FFPPKeypointTracker
from .ffpp_webapi_keypoint_tracker import FFPPWebAPIKeypointTracker
from .utils import load_keypoints, resize_keypoints, visualize_tracking_results

__version__ = "1.0.0"
__author__ = "msraig"

__all__ = [
    "KeypointTracker",
    "FFPPKeypointTracker", 
    "FFPPWebAPIKeypointTracker",
    "load_keypoints", 
    "resize_keypoints",
    "visualize_tracking_results"
]
