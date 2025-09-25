#!/usr/bin/env python3
"""
3D Coordinate Estimation using Triangulation

This script estimates 3D coordinates of keypoints using triangulation method
from multiple view configurations with file path inputs.

Usage:
    # Basic usage (shared camera parameters - most common):
    python 3d_coordinates_estimation.py \\
        --reference-keypoints ref_keypoints.json \\
        --reference-pose ref_pose.json \\
        --keypoint-files view1_tracking.json view2_tracking.json view3_tracking.json \\
        --pose-files pose1.json pose2.json pose3.json \\
        --intrinsic-file camera_intrinsic.json \\
        --extrinsic-file hand_eye_calib.json \\
        --output ./results
    Note that the intrinsic file and extrinsic file support single file serving or independent upload
"""

import json
import numpy as np
import cv2
import os
import sys
import time
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def tcp_pose_to_matrix(tcp_pose):
    """
    Convert TCP pose (x, y, z, rx, ry, rz) to 4x4 transformation matrix.
    
    Args:
        tcp_pose: List or array of [x, y, z, rx, ry, rz] where:
                 - x, y, z are translation in meters
                 - rx, ry, rz are rotation angles in radians (Euler angles in ZYX convention)
    
    Returns:
        4x4 transformation matrix
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


# ============================================================================
# CORE TRIANGULATION FUNCTIONS
# ============================================================================

def triangulate_keypoints(points_2d: List[np.ndarray], 
                         projection_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Triangulate 3D keypoints from multiple views using Direct Linear Transform (DLT).
    
    Args:
        points_2d: List of 2D point arrays for each view
        projection_matrices: List of camera projection matrices (K * [R|t])
        
    Returns:
        3D points array (N, 3)
    """
    if len(points_2d) < 2:
        raise ValueError("At least 2 views required for triangulation")
    
    n_points = len(points_2d[0])
    points_3d = np.zeros((n_points, 3))
    
    for i in range(n_points):
        # Collect corresponding points from all views
        view_points = []
        view_proj_matrices = []
        
        for j, (pts_2d, P) in enumerate(zip(points_2d, projection_matrices)):
            if i < len(pts_2d):
                view_points.append(pts_2d[i])
                view_proj_matrices.append(P)
        
        if len(view_points) >= 2:
            # Perform DLT triangulation for this point
            points_3d[i] = triangulate_single_keypoint(view_points, view_proj_matrices)
        else:
            logger.warning(f"Insufficient views for point {i}")
    
    return points_3d


def triangulate_single_keypoint(points_2d: List[np.ndarray], 
                               projection_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Direct Linear Transform (DLT) triangulation for a single keypoint.
    
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


# ============================================================================
# PUBLIC INTERFACE FUNCTIONS
# ============================================================================

def estimate_3d(
    view_configs: List[Dict],
    reference_keypoints_file: str,
    reference_pose_file: str,
    output_file: Optional[str] = None,
    model_path: Optional[str] = None,
    device: str = 'auto'
) -> Dict:
    """
    General interface function for 3D coordinate estimation from file paths.
    
    This function provides a flexible interface that takes multiple view configurations
    and estimates 3D coordinates using triangulation. Each view consists of:
    - Keypoint tracking results (from FFPPKeypointTracker or similar format)
    - Camera intrinsic parameters 
    - Camera extrinsic parameters (hand-eye calibration)
    
    Args:
        view_configs: List of view configuration dictionaries. Each dict should contain:
            - 'keypoint_file': Path to keypoint tracking results JSON file
            - 'pose_file': Path to robot pose JSON file (contains tcp_pose)
            - 'intrinsic_file': Path to camera intrinsic parameters JSON file  
            - 'extrinsic_file': Path to camera extrinsic parameters JSON file
        reference_keypoints_file: Path to reference keypoints JSON file
        reference_pose_file: Path to reference robot pose JSON file
            
            Example keypoint_file format (from FFPPKeypointTracker.track_keypoints()):
            {
                "success": true,
                "tracked_keypoints": [
                    {"x": 100.5, "y": 200.3, "id": 1, "name": "corner_1"},
                    {"x": 150.2, "y": 180.7, "id": 2, "name": "corner_2"}
                ],
                "reference_keypoints": [...],  # Original keypoints from reference
                ...
            }
            
            Example intrinsic_file format:
            {
                "success": true,
                "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                "distortion_coefficients": [k1, k2, p1, p2, k3]
            }
            
            Example extrinsic_file format:
            {
                "success": true,
                "cam2end_matrix": [[r11, r12, r13, tx], [r21, r22, r23, ty], 
                                   [r31, r32, r33, tz], [0, 0, 0, 1]],
                "target2base_matrix": [[...], [...], [...], [0, 0, 0, 1]]
            }
            
        output_file: Optional output filename for results. If None, generates timestamp-based name.
        model_path: Path to FlowFormer++ model (not used in this interface, kept for compatibility)
        device: Device for computation ('cpu', 'cuda', 'auto')
        
    Returns:
        Dict containing estimation results:
        {
            "success": bool,
            "num_keypoints": int,
            "num_views": int,
            "keypoints_3d": [
                {
                    "id": int,
                    "name": str,
                    "coordinates_3d": {"x": float, "y": float, "z": float},
                    "reference_2d": {"x": float, "y": float}
                }
            ],
            "view_configurations": [...],
            "processing_time": float,
            "output_file": str
        }
        
    Raises:
        ValueError: If input validation fails
        FileNotFoundError: If required files are missing
        RuntimeError: If estimation process fails
        
    Example usage:
        view_configs = [
            {
                'keypoint_file': 'view1_keypoints.json',
                'intrinsic_file': 'camera_intrinsic.json', 
                'extrinsic_file': 'hand_eye_calibration.json'
            },
            {
                'keypoint_file': 'view2_keypoints.json',
                'intrinsic_file': 'camera_intrinsic.json',  # Same camera
                'extrinsic_file': 'hand_eye_calibration.json'  # Same setup
            },
            # Add more views as needed
        ]
        
        results = estimate_3d(view_configs)
    """
    import time
    
    start_time = time.time()
    
    try:
        # Input validation
        if not view_configs or len(view_configs) < 2:
            raise ValueError("At least 2 views are required for triangulation")
        
        logger.info(f"Starting 3D coordinate estimation from {len(view_configs)} views")
        
        # ========================================
        # LOAD AND VALIDATE DATA
        # ========================================
        
        # Load reference keypoints
        logger.info(f"Loading reference keypoints from: {reference_keypoints_file}")
        with open(reference_keypoints_file, 'r') as f:
            reference_data = json.load(f)
        
        if 'keypoints' in reference_data:
            reference_keypoints = reference_data['keypoints']
        elif 'reference_keypoints' in reference_data:
            reference_keypoints = reference_data['reference_keypoints']
        else:
            raise ValueError(f"No 'keypoints' or 'reference_keypoints' found in reference file: {reference_keypoints_file}")
        
        logger.info(f"Loaded {len(reference_keypoints)} reference keypoints")
        
        # Load and validate all keypoint files
        keypoint_data_list = []
        
        for i, config in enumerate(view_configs):
            # Validate required keys
            required_keys = ['keypoint_file', 'pose_file', 'intrinsic_file', 'extrinsic_file']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"View {i}: Missing required key '{key}'")
            
            # Load keypoint file
            keypoint_path = Path(config['keypoint_file'])
            if not keypoint_path.exists():
                raise FileNotFoundError(f"Keypoint file not found: {keypoint_path}")
            
            with open(keypoint_path, 'r') as f:
                keypoint_data = json.load(f)
            
            # Handle nested tracking result format
            if 'tracking_result' in keypoint_data:
                # Nested format: {"image_file": "...", "tracking_result": {"success": ..., "tracked_keypoints": ...}}
                tracking_result = keypoint_data['tracking_result']
                if not tracking_result.get('success', False):
                    raise ValueError(f"View {i}: Keypoint tracking was not successful")
                
                if 'tracked_keypoints' not in tracking_result:
                    raise ValueError(f"View {i}: No 'tracked_keypoints' found in tracking result")
                
                # Replace keypoint_data with the actual tracking result for consistency
                keypoint_data = tracking_result
            else:
                # Direct format: {"success": ..., "tracked_keypoints": ...}
                if not keypoint_data.get('success', False):
                    raise ValueError(f"View {i}: Keypoint tracking was not successful")
                
                if 'tracked_keypoints' not in keypoint_data:
                    raise ValueError(f"View {i}: No 'tracked_keypoints' found in keypoint file")
            
            keypoint_data_list.append(keypoint_data)
        
        # Load camera parameters (support shared intrinsic parameters)
        intrinsic_files = list(set(config['intrinsic_file'] for config in view_configs))
        extrinsic_files = list(set(config['extrinsic_file'] for config in view_configs))
        
        # Cache camera parameters to avoid repeated loading
        intrinsic_cache = {}
        extrinsic_cache = {}
        
        # Load intrinsic parameters
        for intrinsic_file in intrinsic_files:
            intrinsic_path = Path(intrinsic_file)
            if not intrinsic_path.exists():
                raise FileNotFoundError(f"Intrinsic parameters file not found: {intrinsic_path}")
            
            with open(intrinsic_path, 'r') as f:
                intrinsic_data = json.load(f)
            
            if not intrinsic_data.get('success', False):
                raise ValueError(f"Camera calibration was not successful: {intrinsic_file}")
            
            intrinsic_cache[intrinsic_file] = intrinsic_data
        
        # Load extrinsic parameters
        for extrinsic_file in extrinsic_files:
            extrinsic_path = Path(extrinsic_file)
            if not extrinsic_path.exists():
                raise FileNotFoundError(f"Extrinsic parameters file not found: {extrinsic_path}")
            
            with open(extrinsic_path, 'r') as f:
                extrinsic_data = json.load(f)
            
            if not extrinsic_data.get('success', False):
                raise ValueError(f"Hand-eye calibration was not successful: {extrinsic_file}")
            
            extrinsic_cache[extrinsic_file] = extrinsic_data
        
        logger.info(f"Loaded {len(intrinsic_cache)} intrinsic and {len(extrinsic_cache)} extrinsic parameter files")
        
        # ========================================
        # SETUP TRIANGULATION COMPONENTS
        # ========================================
        
        # Process each view to extract 2D points and camera poses
        all_points_2d = []
        all_camera_poses = []
        
        for i, (config, keypoint_data) in enumerate(zip(view_configs, keypoint_data_list)):
            # Get cached parameters
            intrinsic_data = intrinsic_cache[config['intrinsic_file']]
            extrinsic_data = extrinsic_cache[config['extrinsic_file']]
            
            # Extract camera parameters directly
            intrinsic_matrix = np.array(intrinsic_data['camera_matrix'])
            distortion_coeffs = np.array(intrinsic_data['distortion_coefficients'])
            
            # Extract 2D points from keypoint data
            tracked_keypoints = keypoint_data['tracked_keypoints']
            points_2d = np.array([[kp['x'], kp['y']] for kp in tracked_keypoints], dtype=np.float32)
            
            # Undistort points
            points_reshaped = points_2d.reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(
                points_reshaped,
                intrinsic_matrix,
                distortion_coeffs,
                P=intrinsic_matrix
            )
            points_2d_undistorted = undistorted.reshape(-1, 2)
            
            all_points_2d.append(points_2d_undistorted)
            
            # Calculate camera pose using robot pose and hand-eye calibration
            # Load robot pose for this view
            pose_file = config['pose_file']
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
            
            # Extract TCP pose and convert to end2base matrix
            tcp_pose = pose_data['tcp_pose']
            end2base_matrix = tcp_pose_to_matrix(tcp_pose)
            
            # Extract hand-eye calibration matrices
            cam2end_matrix = np.array(extrinsic_data['cam2end_matrix'])
            
            # Calculate camera pose in base coordinate system
            # cam2base = end2base * cam2end
            cam2base_matrix = end2base_matrix @ cam2end_matrix
            
            all_camera_poses.append(cam2base_matrix)
            
            logger.info(f"View {i+1}: Loaded robot pose from {pose_file}")
            logger.debug(f"View {i+1}: TCP pose = {tcp_pose}")
            logger.debug(f"View {i+1}: cam2base matrix = \n{cam2base_matrix}")
        
        # ========================================
        # PERFORM TRIANGULATION
        # ========================================
        
        # Prepare projection matrices for triangulation
        # Use the first view's intrinsic parameters as the reference
        first_intrinsic = intrinsic_cache[view_configs[0]['intrinsic_file']]
        K = np.array(first_intrinsic['camera_matrix'])
        
        projection_matrices = []
        
        for i, pose in enumerate(all_camera_poses):
            # Convert pose to projection matrix: P = K * [R|t]
            # pose is cam2base, we need base2cam for projection
            base2cam_matrix = np.linalg.inv(pose)
            R = base2cam_matrix[:3, :3]
            t = base2cam_matrix[:3, 3]
            RT = np.hstack([R, t.reshape(-1, 1)])
            P = K @ RT
            projection_matrices.append(P)
        
        # Perform triangulation using DLT
        points_3d = triangulate_keypoints(all_points_2d, projection_matrices)
        
        # ========================================
        # PREPARE RESULTS
        # ========================================
        
        processing_time = time.time() - start_time
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = "3d_coordinates_estimation_result.json"
        else:
            # If output_file is provided, treat it as directory path
            # Generate fixed filename in the specified directory
            output_dir = Path(output_file)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "3d_coordinates_estimation_result.json"
            output_file = str(output_file)
        
        results = {
            'success': True,
            'num_keypoints': len(reference_keypoints),
            'num_views': len(view_configs),
            'processing_time': processing_time,
            'keypoints_3d': [],
            'reference_keypoints': reference_keypoints,
            'view_configurations': view_configs,
            'points_2d_all_views': [pts.tolist() for pts in all_points_2d],
            'camera_poses': [pose.tolist() for pose in all_camera_poses],
            'interface_version': 'file_based_v1.0',
            'estimation_method': 'triangulation_dlt',
            'output_file': output_file
        }
        
        # Add 3D coordinates for each keypoint
        for i, (ref_kp, pt_3d) in enumerate(zip(reference_keypoints, points_3d)):
            keypoint_result = {
                'id': ref_kp.get('id', i + 1),
                'name': ref_kp.get('name', f'point_{i + 1}'),
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
        
        # Save results
        save_results_to_file(results, output_file)
        
        logger.info(f"3D coordinate estimation completed successfully in {processing_time:.3f}s")
        logger.info(f"Estimated 3D coordinates for {len(points_3d)} keypoints from {len(view_configs)} views")
        
        return results
        
    except Exception as e:
        logger.error(f"3D coordinate estimation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time,
            'num_views': len(view_configs) if view_configs else 0
        }


def save_results_to_file(results: Dict, filename: str):
    """
    Save estimation results to file.
    
    Args:
        results: Results dictionary to save
        filename: Output filename (can be absolute path or relative to project root)
    """
    # Convert to Path object for easier handling
    output_path = Path(filename)
    
    # If it's not an absolute path, make it relative to project root
    if not output_path.is_absolute():
        project_root = Path(__file__).parent.parent
        output_path = project_root / filename
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    if results.get('success', False):
        print("\n" + "="*70)
        print("3D COORDINATE ESTIMATION RESULTS")
        print("="*70)
        print(f"Interface version: {results.get('interface_version', 'unknown')}")
        print(f"Estimation method: {results.get('estimation_method', 'unknown')}")
        print(f"Number of keypoints: {results['num_keypoints']}")
        print(f"Number of views used: {results['num_views']}")
        print(f"Processing time: {results.get('processing_time', 0):.3f}s")
        print(f"Output file: {output_path}")
        print("\n3D Coordinates Result:")
        print("-" * 70)
        
        for kp in results['keypoints_3d']:
            coords = kp['coordinates_3d']
            print(f"Keypoint {kp['id']} ({kp['name']}):")
            print(f"  X: {coords['x']:.6f} m")
            print(f"  Y: {coords['y']:.6f} m")
            print(f"  Z: {coords['z']:.6f} m")
            print()
        
        print("="*70)
    else:
        print(f"\n❌ 3D coordinate estimation failed: {results.get('error', 'Unknown error')}")


def parse_arguments():
    """Parse command line arguments for 3D coordinate estimation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="3D Coordinate Estimation using Triangulation from File Paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:

Basic usage with shared camera parameters (most common):
python 3d_coordinates_estimation.py \\
    --reference-keypoints ref_keypoints.json \\
    --reference-pose ref_pose.json \\
    --keypoint-files view1_tracking.json view2_tracking.json view3_tracking.json \\
    --pose-files pose1.json pose2.json pose3.json \\
    --intrinsic-file camera_intrinsic.json \\
    --extrinsic-file hand_eye_calib.json \\
    --output ./results

For different camera parameters, use --intrinsic-files and/or --extrinsic-files instead.

Important Notes:
- At least 2 views required for triangulation
- Pose files contain robot TCP pose: [x, y, z, rx, ry, rz] 
- Camera parameters can be shared (one file) or individual (multiple files)
        """
    )
    
    # Required arguments - Reference files come first
    parser.add_argument(
        '--reference-keypoints',
        required=True,
        help='Path to reference keypoints file (contains manually labeled reference points)'
    )
    parser.add_argument(
        '--reference-pose',
        required=True,
        help='Path to reference robot pose file (robot pose when reference image was taken)'
    )
    parser.add_argument(
        '--keypoint-files',
        nargs='+',
        required=True,
        help='Paths to keypoint tracking result files (FFPPKeypointTracker output)'
    )
    parser.add_argument(
        '--pose-files',
        nargs='+',
        required=True,
        help='Paths to robot pose files for each view (one per keypoint file)'
    )
    
    # Extrinsic parameters (choose one)
    extrinsic_group = parser.add_mutually_exclusive_group(required=True)
    extrinsic_group.add_argument(
        '--extrinsic-file',
        help='Path to camera extrinsic parameters file (shared for all views - same hand-eye calibration)'
    )
    extrinsic_group.add_argument(
        '--extrinsic-files',
        nargs='+',
        help='Paths to individual extrinsic parameter files for each view (different hand-eye calibrations)'
    )
    
    # Intrinsic parameters (choose one)
    intrinsic_group = parser.add_mutually_exclusive_group(required=True)
    intrinsic_group.add_argument(
        '--intrinsic-file',
        help='Path to camera intrinsic parameters file (shared for all views)'
    )
    intrinsic_group.add_argument(
        '--intrinsic-files',
        nargs='+',
        help='Paths to individual intrinsic parameter files for each view'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output',
        help='Output directory path for saving all results (optional, uses default output directory if not provided)'
    )
    
    return parser.parse_args()


def validate_interface_args(args):
    """Validate arguments for 3D coordinate estimation."""
    if len(args.keypoint_files) < 2:
        raise ValueError("At least 2 keypoint files are required for triangulation")
    
    # Validate pose files count
    if len(args.pose_files) != len(args.keypoint_files):
        raise ValueError(f"Number of pose files ({len(args.pose_files)}) must match "
                        f"number of keypoint files ({len(args.keypoint_files)})")
    
    # Validate extrinsic parameters count
    if args.extrinsic_files:
        if len(args.extrinsic_files) != len(args.keypoint_files):
            raise ValueError(f"Number of extrinsic files ({len(args.extrinsic_files)}) must match "
                            f"number of keypoint files ({len(args.keypoint_files)})")
    
    # Validate intrinsic parameters count if individual files specified
    if args.intrinsic_files:
        if len(args.intrinsic_files) != len(args.keypoint_files):
            raise ValueError(f"Number of intrinsic files ({len(args.intrinsic_files)}) must match "
                            f"number of keypoint files ({len(args.keypoint_files)})")
    
    # Check file existence
    all_files = [args.reference_keypoints, args.reference_pose] + args.keypoint_files + args.pose_files
    if args.intrinsic_file:
        all_files.append(args.intrinsic_file)
    if args.intrinsic_files:
        all_files.extend(args.intrinsic_files)
    if args.extrinsic_file:
        all_files.append(args.extrinsic_file)
    if args.extrinsic_files:
        all_files.extend(args.extrinsic_files)
    
    for file_path in all_files:
        if not Path(file_path).exists():
            raise ValueError(f"File not found: {file_path}")
    
    logger.info(f"Validated {len(args.keypoint_files)} views for triangulation")
    
    # Log the parameter combination being used
    intrinsic_mode = "shared" if args.intrinsic_file else "individual"
    extrinsic_mode = "shared" if args.extrinsic_file else "individual"
    print(f"📋 Parameter combination: {intrinsic_mode} intrinsic + {extrinsic_mode} extrinsic + individual pose (required for triangulation)")


def run_estimation(args):
    """Run 3D coordinate estimation from file paths."""
    # Validate arguments
    validate_interface_args(args)
    
    # Build view configurations
    view_configs = []
    num_views = len(args.keypoint_files)
    
    for i in range(num_views):
        # Keypoint file
        keypoint_file = args.keypoint_files[i]
        
        # Pose file
        pose_file = args.pose_files[i]
        
        # Intrinsic file
        if args.intrinsic_file:
            intrinsic_file = args.intrinsic_file  # Shared
        else:
            intrinsic_file = args.intrinsic_files[i]  # Individual
        
        # Extrinsic file
        if args.extrinsic_file:
            extrinsic_file = args.extrinsic_file  # Shared
        else:
            extrinsic_file = args.extrinsic_files[i]  # Individual
        
        view_config = {
            'keypoint_file': keypoint_file,
            'pose_file': pose_file,
            'intrinsic_file': intrinsic_file,
            'extrinsic_file': extrinsic_file
        }
        view_configs.append(view_config)
    
    print(f"🚀 Running 3D estimation with {num_views} views...")
    print("📁 Reference files:")
    print(f"   Reference keypoints: {args.reference_keypoints}")
    print(f"   Reference pose:      {args.reference_pose}")
    print("📁 View configurations:")
    for i, config in enumerate(view_configs, 1):
        print(f"   View {i}:")
        print(f"     Keypoints: {config['keypoint_file']}")
        print(f"     Pose:      {config['pose_file']}")
        print(f"     Intrinsic: {config['intrinsic_file']}")
        print(f"     Extrinsic: {config['extrinsic_file']}")
    
    # Run estimation
    results = estimate_3d(
        view_configs=view_configs,
        reference_keypoints_file=args.reference_keypoints,
        reference_pose_file=args.reference_pose,
        output_file=args.output,
        model_path=None,  # Not used in this interface
        device='auto'  # Always use auto device detection
    )
    
    if results['success']:
        print("✅ 3D estimation completed successfully!")
        return 0
    else:
        print(f"❌ 3D estimation failed: {results['error']}")
        return 1



def main():
    """Main function - 3D coordinate estimation from file paths only."""
    try:
        args = parse_arguments()
        return run_estimation(args)
                
    except Exception as e:
        logger.error(f"3D coordinate estimation failed: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())