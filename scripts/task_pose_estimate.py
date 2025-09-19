#!/usr/bin/env python3
"""
3D Positioning Estimation using Triangulation (FFPPKeypointTracker Version)

This script estimates 3D coordinates of keypoints using triangulation method.
It uses multi-view geometry with robot poses and camera calibration data.
This version uses the FFPPKeypointTracker class for improved performance.

Data inputs:
- Reference image: test_data/positioning_data/ref_img.jpg
- Reference keypoints: test_data/positioning_data/ref_label.json
- Reference robot pose: test_data/positioning_data/ref_joints_position.json (includes tcp_pose)
- Camera intrinsics: test_data/camera_parameters/calibration_result.json
- Camera extrinsics: test_data/camera_parameters/eye_in_hand_result.json
- Image sequence: test_data/positioning_data/1-5.jpg
- Robot poses: test_data/positioning_data/1-5.json (includes tcp_pose for accurate end-effector positioning)
"""

import json
import numpy as np
import cv2
import os
import sys
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Add core to path for importing keypoint tracker
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.ffpp_keypoint_tracker import FFPPKeypointTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CameraParameters:
    """Class to handle camera calibration parameters"""
    
    def __init__(self, intrinsic_file: str, extrinsic_file: str):
        """
        Initialize camera parameters from calibration files
        
        Args:
            intrinsic_file: Path to camera intrinsic calibration file
            extrinsic_file: Path to eye-in-hand calibration file
        """
        self.intrinsic_matrix = None
        self.distortion_coeffs = None
        self.cam2end_matrix = None
        self.target2base_matrix = None
        
        self.load_intrinsic_parameters(intrinsic_file)
        self.load_extrinsic_parameters(extrinsic_file)
    
    def load_intrinsic_parameters(self, filename: str):
        """Load camera intrinsic parameters from calibration file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if not data.get('success', False):
                raise ValueError("Calibration was not successful")
            
            self.intrinsic_matrix = np.array(data['camera_matrix'])
            self.distortion_coeffs = np.array(data['distortion_coefficients'])
            
            logger.info(f"Loaded camera intrinsic parameters from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load intrinsic parameters: {e}")
            raise
    
    def load_extrinsic_parameters(self, filename: str):
        """Load camera extrinsic parameters (eye-in-hand calibration)"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if not data.get('success', False):
                raise ValueError("Eye-in-hand calibration was not successful")
            
            self.cam2end_matrix = np.array(data['cam2end_matrix'])
            self.target2base_matrix = np.array(data['target2base_matrix'])

            logger.info(f"Loaded camera extrinsic parameters from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load extrinsic parameters: {e}")
            raise


class CameraPoseCalculator:
    """Class to calculate camera poses using calibration data"""
    
    def __init__(self, camera_params: CameraParameters):
        """
        Initialize camera pose calculator
        
        Args:
            camera_params: Camera calibration parameters containing cam2end and target2base_matrix matrices
        """
        self.camera_params = camera_params
        logger.info("Camera pose calculator initialized")
    
    def tcp_pose_to_matrix(self, tcp_pose: List[float]) -> np.ndarray:
        """
        Convert TCP pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix
        
        Args:
            tcp_pose: TCP pose as [x, y, z, rx, ry, rz] where:
                     - x, y, z are translation coordinates in meters (m)
                     - rx, ry, rz are Euler angles in radians
            
        Returns:
            4x4 transformation matrix with translation in meters
        """
        x, y, z, rx, ry, rz = tcp_pose
        
        # Create rotation matrix from Euler angles (ZYX intrinsic/body-fixed convention)
        # ZYX intrinsic rotation sequence: first Z, then Y, then X
        # rx = final rotation around current X-axis (roll)
        # ry = middle rotation around current Y-axis (pitch) 
        # rz = first rotation around current Z-axis (yaw)
        
        # Individual rotation matrices
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)
        
        # Rotation matrix around x-axis (roll)
        Rx = np.array([[1, 0, 0],
                       [0, cos_rx, -sin_rx],
                       [0, sin_rx, cos_rx]])
        
        # Rotation matrix around y-axis (pitch)
        Ry = np.array([[cos_ry, 0, sin_ry],
                       [0, 1, 0],
                       [-sin_ry, 0, cos_ry]])
        
        # Rotation matrix around z-axis (yaw)
        Rz = np.array([[cos_rz, -sin_rz, 0],
                       [sin_rz, cos_rz, 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix (ZYX convention: R = Rz * Ry * Rx)
        rotation_matrix = Rz @ Ry @ Rx
        
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = [x, y, z]
        
        return transform_matrix
    
    def get_camera_pose_from_tcp(self, tcp_pose: List[float]) -> np.ndarray:
        """
        Calculate camera pose from TCP pose using calibration data
        
        Formula: T_cam_to_world = T_end_to_world × T_cam_to_end
        
        Args:
            tcp_pose: TCP pose as [x, y, z, rx, ry, rz]
            
        Returns:
            4x4 transformation matrix from camera to world (T_cam_to_world)
        """
        # Convert TCP pose to transformation matrix
        end2base_matrix = self.tcp_pose_to_matrix(tcp_pose)
        
        # Apply camera-to-end-effector transformation
        # T_cam2base = T_end2base × T_cam2end
        cam2base_matrix = end2base_matrix @ self.camera_params.cam2end_matrix
        
        return cam2base_matrix
    

class KeypointMatcher:
    """Class to handle keypoint detection and matching using FFPPKeypointTracker"""
    
    def __init__(self, camera_params: CameraParameters, data_dir: str = "test_data", 
                 model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize keypoint matcher with FFPPKeypointTracker
        
        Args:
            camera_params: Camera calibration parameters
            data_dir: Base directory for saving visualization results
            model_path: Path to FlowFormer++ model checkpoint
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.camera_params = camera_params
        self.data_dir = Path(data_dir)
        
        # Initialize FFPPKeypointTracker instead of KeypointTracker
        self.tracker = FFPPKeypointTracker(
            model_path=model_path,
            device=device,
            max_image_size=1024
        )
        
        logger.info(f"Initialized FFPPKeypointTracker with device: {device}")
    
    def convert_image_bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB format for FFPPKeypointTracker"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def track_keypoints_with_optical_flow(self, ref_image: np.ndarray, target_image: np.ndarray,
                                        ref_points: List[Dict], save_visualization: bool = True, 
                                        sequence_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track keypoints using FFPPKeypointTracker
        
        Args:
            ref_image: Reference image (BGR format from OpenCV)
            target_image: Target image (BGR format from OpenCV)
            ref_points: Reference keypoint coordinates
            save_visualization: Whether to save tracking visualization
            sequence_idx: Sequence index for filename generation
            
        Returns:
            Tuple of (reference_points, target_points) as numpy arrays
        """
        # Convert reference points to numpy array
        ref_pts = np.array([[pt['x'], pt['y']] for pt in ref_points], dtype=np.float32)
        
        # Convert BGR images to RGB for FFPPKeypointTracker
        ref_image_rgb = self.convert_image_bgr_to_rgb(ref_image)
        target_image_rgb = self.convert_image_bgr_to_rgb(target_image)
        
        # Convert reference points to the format expected by FFPPKeypointTracker
        tracker_keypoints = []
        for pt in ref_points:
            tracker_kp = {
                'x': pt['x'],
                'y': pt['y']
            }
            # Add optional fields if available
            if 'id' in pt:
                tracker_kp['id'] = pt['id']
            if 'name' in pt:
                tracker_kp['name'] = pt['name']
            tracker_keypoints.append(tracker_kp)
        
        # Set reference image with keypoints
        ref_result = self.tracker.set_reference_image(ref_image_rgb, tracker_keypoints)
        if not ref_result['success']:
            raise RuntimeError(f"Failed to set reference image: {ref_result['error']}")
        
        logger.info(f"Set reference image with {ref_result['keypoints_count']} keypoints")
        
        # Track keypoints in target image
        track_result = self.tracker.track_keypoints(target_image_rgb, bidirectional=False)
        if not track_result['success']:
            raise RuntimeError(f"Failed to track keypoints: {track_result['error']}")
        
        # Extract tracked coordinates
        # Note: FFPPKeypointTracker returns coordinates in 'x', 'y' fields (not 'tracked_x', 'tracked_y')
        tracked_keypoints = track_result['tracked_keypoints']
        target_pts = np.array([[kp['x'], kp['y']] 
                              for kp in tracked_keypoints], dtype=np.float32)
        
        logger.info(f"Successfully tracked {len(target_pts)} keypoints using FFPPKeypointTracker")
        
        # Save visualization if requested
        if save_visualization:
            filename = f"{sequence_idx}_result.jpg"
            self.visualize_tracking_results(ref_image, target_image, ref_pts, target_pts, ref_points, filename)
            logger.info(f"Tracking visualization saved as {filename}")
        
        return ref_pts, target_pts

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Undistort image points using camera parameters
        
        Args:
            points: Distorted image points
            
        Returns:
            Undistorted image points
        """
        points_reshaped = points.reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(
            points_reshaped,
            self.camera_params.intrinsic_matrix,
            self.camera_params.distortion_coeffs,
            P=self.camera_params.intrinsic_matrix
        )
        return undistorted.reshape(-1, 2)
    
    def visualize_tracking_results(self, ref_image: np.ndarray, target_image: np.ndarray,
                                 ref_pts: np.ndarray, target_pts: np.ndarray, 
                                 ref_points_dict: List[Dict], filename: str = "0_result.jpg"):
        """
        Visualize keypoint tracking results and save to file
        
        Args:
            ref_image: Reference image
            target_image: Target image  
            ref_pts: Reference keypoint coordinates
            target_pts: Tracked keypoint coordinates
            ref_points_dict: Original reference points with names/IDs
            filename: Output filename
        """
        # Create side-by-side visualization
        h1, w1 = ref_image.shape[:2]
        h2, w2 = target_image.shape[:2]
        
        # Make sure both images have same height for side-by-side display
        max_height = max(h1, h2)
        
        # Create canvas
        canvas = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)
        
        # Place images on canvas
        canvas[:h1, :w1] = ref_image
        canvas[:h2, w1:w1+w2] = target_image
        
        # Define colors for different keypoints
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        # Draw keypoints only (no connection lines)
        for i, (ref_pt, target_pt) in enumerate(zip(ref_pts, target_pts)):
            color = colors[i % len(colors)]
            
            # Draw reference point
            ref_x, ref_y = int(ref_pt[0]), int(ref_pt[1])
            cv2.circle(canvas, (ref_x, ref_y), 8, color, 2)
            
            # Draw tracked point (offset by image width)
            target_x, target_y = int(target_pt[0] + w1), int(target_pt[1])
            cv2.circle(canvas, (target_x, target_y), 8, color, 2)
            
            # Add point labels
            if i < len(ref_points_dict):
                point_name = ref_points_dict[i].get('name', f'Point {i+1}')
                cv2.putText(canvas, point_name, (ref_x-20, ref_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(canvas, point_name, (target_x-20, target_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add title
        cv2.putText(canvas, "Reference Image", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Target Image (Tracked)", (w1 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization to robot_vision/output/pose_estimate_output directory
        project_root = Path(__file__).parent.parent  # Go up from scripts/ to robot_vision/
        output_path = project_root / "output" / "pose_estimate_output" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), canvas)
        logger.info(f"Tracking visualization saved to: {output_path}")


class Triangulator:
    """Class to perform 3D triangulation"""
    
    def __init__(self, camera_params: CameraParameters):
        """
        Initialize triangulator
        
        Args:
            camera_params: Camera calibration parameters
        """
        self.camera_params = camera_params
    
    def triangulate_points(self, points_2d: List[np.ndarray], 
                          camera_poses: List[np.ndarray]) -> np.ndarray:
        """
        Triangulate 3D points from multiple views
        
        Args:
            points_2d: List of 2D point arrays for each view
            camera_poses: List of camera pose matrices
            
        Returns:
            3D points in base coordinate system
        """
        if len(points_2d) < 2:
            raise ValueError("At least 2 views required for triangulation")
        
        n_points = len(points_2d[0])
        points_3d = np.zeros((n_points, 3))
        
        # Get camera projection matrices
        K = self.camera_params.intrinsic_matrix
        
        for i in range(n_points):
            # Collect corresponding points from all views
            view_points = []
            projection_matrices = []
            
            # pose here is cam2base
            for j, (pts_2d, pose) in enumerate(zip(points_2d, camera_poses)):
                if i < len(pts_2d):
                    view_points.append(pts_2d[i])
                    
                    base2cam_matrix = np.linalg.inv(pose)
                    R = base2cam_matrix[:3, :3]
                    t = base2cam_matrix[:3, 3]
                    RT = np.hstack([R, t.reshape(-1, 1)])
                    P = K @ RT
                    projection_matrices.append(P)
            
            if len(view_points) >= 2:
                # Perform DLT triangulation
                points_3d[i] = self.dlt_triangulate(view_points, projection_matrices)
            else:
                logger.warning(f"Insufficient views for point {i}")
        
        return points_3d
    
    def dlt_triangulate(self, points_2d: List[np.ndarray], 
                       projection_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Direct Linear Transform (DLT) triangulation
        
        Args:
            points_2d: 2D points from different views
            projection_matrices: Corresponding projection matrices
            
        Returns:
            3D point coordinates
        """
        n_views = len(points_2d)
        A = np.zeros((2 * n_views, 4))
        
        for i, (pt_2d, P) in enumerate(zip(points_2d, projection_matrices)):
            x, y = pt_2d[0], pt_2d[1]
            
            # Build linear system Ax = 0
            A[2*i] = x * P[2] - P[0]
            A[2*i + 1] = y * P[2] - P[1]
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        # Convert from homogeneous coordinates
        if abs(X[3]) > 1e-6:
            return X[:3] / X[3]
        else:
            logger.warning("Triangulation resulted in point at infinity")
            return X[:3]


class PositioningEstimator:
    """Main class for 3D positioning estimation using FFPPKeypointTracker"""
    
    def __init__(self, data_dir: str = "test_data", model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the positioning estimator
        
        Args:
            data_dir: Base directory containing test data
            model_path: Path to FlowFormer++ model checkpoint
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.data_dir = Path(data_dir)
        
        # Initialize camera parameters
        intrinsic_file = self.data_dir / "camera_parameters" / "calibration_result.json"
        extrinsic_file = self.data_dir / "camera_parameters" / "eye_in_hand_result.json"
        
        self.camera_params = CameraParameters(str(intrinsic_file), str(extrinsic_file))
        
        # Initialize components with FFPPKeypointTracker
        self.camera_pose_calculator = CameraPoseCalculator(self.camera_params)
        self.keypoint_matcher = KeypointMatcher(
            self.camera_params, 
            str(self.data_dir),
            model_path=model_path,
            device=device
        )
        self.triangulator = Triangulator(self.camera_params)
        
    def load_reference_data(self) -> Tuple[np.ndarray, List[Dict], List[float]]:
        """
        Load reference image, keypoints, and TCP pose
        
        Returns:
            Tuple of (reference_image, reference_keypoints, reference_tcp_pose)
        """
        # Load reference image
        ref_img_path = self.data_dir / "positioning_data" / "ref_img.jpg"
        ref_image = cv2.imread(str(ref_img_path))
        if ref_image is None:
            raise FileNotFoundError(f"Reference image not found: {ref_img_path}")
        
        # Load reference keypoints
        ref_label_path = self.data_dir / "positioning_data" / "ref_label.json"
        with open(ref_label_path, 'r') as f:
            ref_data = json.load(f)
        ref_keypoints = ref_data['keypoints']
        
        # Load reference TCP pose
        ref_pose_path = self.data_dir / "positioning_data" / "ref_joints_position.json"
        with open(ref_pose_path, 'r') as f:
            ref_pose_data = json.load(f)
        ref_tcp_pose = ref_pose_data.get('tcp_pose')
        
        if ref_tcp_pose is None:
            raise ValueError("TCP pose is required but not found in reference data")
            
        logger.info(f"Loaded reference data: {len(ref_keypoints)} keypoints with TCP pose")
        
        return ref_image, ref_keypoints, ref_tcp_pose
    
    def load_sequence_data(self, sequence_indices: List[int]) -> Tuple[List[np.ndarray], List[List[float]]]:
        """
        Load image sequence and corresponding TCP poses
        
        Args:
            sequence_indices: List of sequence indices to load (e.g., [1, 2, 3, 4, 5])
            
        Returns:
            Tuple of (images, tcp_poses_list)
        """
        images = []
        tcp_poses_list = []
        
        for idx in sequence_indices:
            # Load image
            img_path = self.data_dir / "positioning_data" / f"{idx}.jpg"
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not load image: {img_path}")
                continue
            
            # Load TCP pose
            pose_path = self.data_dir / "positioning_data" / f"{idx}.json"
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            
            tcp_pose = pose_data.get('tcp_pose')
            if tcp_pose is None:
                raise ValueError(f"TCP pose is required but not found in {pose_path}")
            
            images.append(image)
            tcp_poses_list.append(tcp_pose)
        
        logger.info(f"Loaded sequence data: {len(images)} images with TCP poses")
        
        return images, tcp_poses_list
    
    def estimate_3d_positions(self, sequence_indices: Optional[List[int]] = None) -> Dict:
        """
        Perform complete 3D position estimation using FFPPKeypointTracker
        
        Args:
            sequence_indices: List of sequence indices to use (default: [1,2,3,4,5])
            
        Returns:
            Dictionary containing estimation results
        """
        if sequence_indices is None:
            sequence_indices = list(range(1, 6))  # Default to 1-5 (matching available files)
        
        logger.info("Starting 3D position estimation with FFPPKeypointTracker...")
        
        # Load reference data
        ref_image, ref_keypoints, ref_tcp_pose = self.load_reference_data()
        
        # Load sequence data
        sequence_images, sequence_tcp_poses = self.load_sequence_data(sequence_indices)
        
        # Match keypoints in all images
        all_points_2d = []
        all_camera_poses = []
        
        # Add reference view - use TCP pose for accurate camera positioning
        ref_camera_pose = self.camera_pose_calculator.get_camera_pose_from_tcp(ref_tcp_pose)
            
        ref_pts = np.array([[pt['x'], pt['y']] for pt in ref_keypoints], dtype=np.float32)
        ref_pts_undistorted = self.keypoint_matcher.undistort_points(ref_pts)
        
        all_points_2d.append(ref_pts_undistorted)
        all_camera_poses.append(ref_camera_pose)
        
        # Process sequence images
        for i, (seq_img, seq_tcp_pose) in enumerate(zip(sequence_images, sequence_tcp_poses)):
            # Track keypoints using FFPPKeypointTracker
            ref_pts, target_pts = self.keypoint_matcher.track_keypoints_with_optical_flow(
                ref_image, seq_img, ref_keypoints, save_visualization=True, sequence_idx=sequence_indices[i]
            )
            
            # Undistort points
            target_pts_undistorted = self.keypoint_matcher.undistort_points(target_pts)
            
            # Calculate camera pose using TCP pose
            camera_pose = self.camera_pose_calculator.get_camera_pose_from_tcp(seq_tcp_pose)
            logger.debug(f"Using TCP pose for sequence image {sequence_indices[i]}")
            
            all_points_2d.append(target_pts_undistorted)
            all_camera_poses.append(camera_pose)
        
        # Perform triangulation
        points_3d = self.triangulator.triangulate_points(all_points_2d, all_camera_poses)
        
        # Prepare results
        results = {
            'success': True,
            'num_keypoints': len(ref_keypoints),
            'num_views': len(all_points_2d),
            'sequence_indices': sequence_indices,
            'keypoints_3d': [],
            'reference_keypoints': ref_keypoints,
            'camera_poses': [pose.tolist() for pose in all_camera_poses],
            'points_2d_all_views': [pts.tolist() for pts in all_points_2d],
            'tcp_poses_used': {
                'reference': True,
                'sequence': [True] * len(sequence_tcp_poses)
            },
            'tracker_type': 'FFPPKeypointTracker'
        }
        
        # Add 3D coordinates for each keypoint
        for i, (ref_kp, pt_3d) in enumerate(zip(ref_keypoints, points_3d)):
            keypoint_result = {
                'id': ref_kp['id'],
                'name': ref_kp['name'],
                'coordinates_3d': {
                    'x': float(pt_3d[0]),
                    'y': float(pt_3d[1]),
                    'z': float(pt_3d[2])
                },
                'reference_2d': {
                    'x': ref_kp['x'],
                    'y': ref_kp['y']
                }
            }
            results['keypoints_3d'].append(keypoint_result)
        
        logger.info(f"3D estimation completed successfully for {len(points_3d)} keypoints")
        return results
    
    def save_results(self, results: Dict, output_file: str = "keypoints_3dposition_estimated_results.json"):
        """
        Save estimation results to file
        
        Args:
            results: Results dictionary from estimate_3d_positions
            output_file: Output filename
        """
        # Save to robot_vision/output/pose_estimate_output directory
        project_root = Path(__file__).parent.parent  # Go up from scripts/ to robot_vision/
        output_path = project_root / "output" / "pose_estimate_output" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("3D POSITIONING ESTIMATION RESULTS (FFPPKeypointTracker)")
        print("="*60)
        print(f"Tracker type: {results.get('tracker_type', 'FFPPKeypointTracker')}")
        print(f"Number of keypoints: {results['num_keypoints']}")
        print(f"Number of views used: {results['num_views']}")
        print("\n3D Coordinates Result:")
        print("-" * 60)
        
        for kp in results['keypoints_3d']:
            coords = kp['coordinates_3d']
            print(f"Keypoint {kp['id']} ({kp['name']}):")
            print(f"  X: {coords['x']:.6f} m")
            print(f"  Y: {coords['y']:.6f} m")
            print(f"  Z: {coords['z']:.6f} m")
            print()


def main():
    """Main function to run 3D positioning estimation with FFPPKeypointTracker"""
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        
        # Initialize estimator with FFPPKeypointTracker using scripts directory
        # You can specify model_path and device here if needed
        estimator = PositioningEstimator(
            data_dir=str(script_dir),  # Use scripts directory as data_dir
            model_path=None,  # Uses default sintel.pth
            device='auto'     # Auto selects GPU if available
        )
        
        # Perform estimation using all available sequence images (1-5)
        results = estimator.estimate_3d_positions(sequence_indices=[1, 2, 3, 4, 5])
        
        # Save results with FFPP-specific filename
        estimator.save_results(results, "ffpp_positioning_results.json")
        
        print("3D positioning estimation with FFPPKeypointTracker completed successfully!")
        
    except Exception as e:
        logger.error(f"Estimation failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())