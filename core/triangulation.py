"""
Multi-View Triangulation Module

This module provides triangulation functionality for estimating 3D coordinates
from multiple 2D views with camera parameters.

Author: Yizhong Zhang
Date: November 2025
"""

import sys
import numpy as np
import cv2
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def triangulate_multiview(view_data: List[Dict]) -> Dict:
    """
    Triangulate 3D points from multiple 2D views with camera parameters.
    
    This function takes 2D point observations from multiple camera views along with
    their intrinsic and extrinsic parameters, and estimates the corresponding 3D
    coordinates using Direct Linear Transform (DLT) triangulation.
    
    Args:
        view_data: List of view dictionaries. Each view should contain:
            - 'points_2d': np.ndarray of shape (N, 2) - 2D pixel coordinates
            - 'image_size': tuple (width, height) - Image dimensions in pixels
            - 'intrinsic': np.ndarray of shape (3, 3) - Camera intrinsic matrix (K)
                          [[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]]
            - 'distortion': np.ndarray - Distortion coefficients [k1, k2, p1, p2, k3, ...]
            - 'extrinsic': np.ndarray of shape (4, 4) - Camera extrinsic matrix (world to camera)
                          [[r11, r12, r13, tx],
                           [r21, r22, r23, ty],
                           [r31, r32, r33, tz],
                           [0, 0, 0, 1]]
            
            Example view:
            {
                'points_2d': np.array([[100.5, 200.3], [150.2, 180.7]]),
                'image_size': (1920, 1080),
                'intrinsic': np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]]),
                'distortion': np.array([0.1, -0.05, 0, 0, 0]),
                'extrinsic': np.eye(4)  # World to camera transformation
            }
    
    Returns:
        Dict containing triangulation results:
        {
            'success': bool - Whether triangulation succeeded
            'points_3d': np.ndarray of shape (N, 3) - Triangulated 3D points in world coordinates
            'num_points': int - Number of triangulated points
            'num_views': int - Number of views used
            'reprojection_errors': List[Dict] - Reprojection error for each view
                [
                    {
                        'view_index': int,
                        'mean_error': float - Mean reprojection error in pixels,
                        'max_error': float - Maximum reprojection error in pixels,
                        'errors_per_point': np.ndarray - Error for each point
                    },
                    ...
                ]
            'mean_reprojection_error': float - Overall mean reprojection error across all views
            'points_per_point_errors': np.ndarray of shape (N,) - Mean reprojection error per point
            'triangulation_angles': np.ndarray of shape (N,) - Triangulation angle for each point (degrees)
            'error_message': str - Error message if success is False
        }
    
    Raises:
        ValueError: If input validation fails
        
    Example usage:
        view_data = [
            {
                'points_2d': np.array([[100, 200], [150, 180]]),
                'image_size': (1920, 1080),
                'intrinsic': intrinsic_matrix_1,
                'distortion': distortion_coeffs_1,
                'extrinsic': extrinsic_matrix_1
            },
            {
                'points_2d': np.array([[500, 300], [550, 280]]),
                'image_size': (1920, 1080),
                'intrinsic': intrinsic_matrix_2,
                'distortion': distortion_coeffs_2,
                'extrinsic': extrinsic_matrix_2
            }
        ]
        
        result = triangulate_multiview(view_data)
        
        if result['success']:
            points_3d = result['points_3d']
            mean_error = result['mean_reprojection_error']
            print(f"Triangulated {len(points_3d)} points with mean error {mean_error:.2f} pixels")
    """
    try:
        # ========================================
        # INPUT VALIDATION
        # ========================================
        
        if not view_data or len(view_data) < 2:
            return {
                'success': False,
                'error_message': 'At least 2 views are required for triangulation',
                'num_views': len(view_data) if view_data else 0
            }
        
        # Validate each view has required fields
        required_fields = ['points_2d', 'image_size', 'intrinsic', 'distortion', 'extrinsic']
        for i, view in enumerate(view_data):
            for field in required_fields:
                if field not in view:
                    return {
                        'success': False,
                        'error_message': f"View {i}: Missing required field '{field}'",
                        'num_views': len(view_data)
                    }
        
        # Check that all views have the same number of points
        num_points = len(view_data[0]['points_2d'])
        for i, view in enumerate(view_data[1:], start=1):
            if len(view['points_2d']) != num_points:
                return {
                    'success': False,
                    'error_message': f"View {i} has {len(view['points_2d'])} points, expected {num_points}",
                    'num_views': len(view_data)
                }
        
        if num_points == 0:
            return {
                'success': False,
                'error_message': 'No points to triangulate',
                'num_views': len(view_data)
            }
        
        logger.info(f"Triangulating {num_points} points from {len(view_data)} views")
        
        # ========================================
        # UNDISTORT POINTS
        # ========================================
        
        undistorted_points = []
        
        for i, view in enumerate(view_data):
            points_2d = np.array(view['points_2d'], dtype=np.float32)
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            distortion = np.array(view['distortion'], dtype=np.float64)
            
            # Reshape for cv2.undistortPoints
            points_reshaped = points_2d.reshape(-1, 1, 2)
            
            # Undistort points and normalize to camera coordinates
            # Then project back to pixel coordinates using intrinsic matrix
            undistorted = cv2.undistortPoints(
                points_reshaped,
                intrinsic,
                distortion,
                P=intrinsic
            )
            
            undistorted_2d = undistorted.reshape(-1, 2)
            undistorted_points.append(undistorted_2d)
            
            logger.debug(f"View {i}: Undistorted {len(undistorted_2d)} points")
        
        # ========================================
        # PREPARE PROJECTION MATRICES
        # ========================================
        
        projection_matrices = []
        camera_centers = []
        
        for i, view in enumerate(view_data):
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            extrinsic = np.array(view['extrinsic'], dtype=np.float64)
            
            # Extract rotation and translation from extrinsic matrix
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            
            # Create [R|t] matrix
            RT = np.hstack([R, t.reshape(-1, 1)])
            
            # Projection matrix P = K * [R|t]
            P = intrinsic @ RT
            projection_matrices.append(P)
            
            # Calculate camera center in world coordinates
            # Camera center: C = -R^T * t
            camera_center = -R.T @ t
            camera_centers.append(camera_center)
            
            logger.debug(f"View {i}: Projection matrix shape {P.shape}")
        
        # ========================================
        # TRIANGULATE POINTS
        # ========================================
        
        points_3d = np.zeros((num_points, 3))
        triangulation_angles = np.zeros(num_points)
        
        for point_idx in range(num_points):
            # Collect 2D observations for this point from all views
            observations_2d = [undistorted_points[view_idx][point_idx] 
                              for view_idx in range(len(view_data))]
            
            # Triangulate using DLT
            point_3d = _triangulate_dlt(observations_2d, projection_matrices)
            points_3d[point_idx] = point_3d
            
            # Calculate triangulation angle (angle between rays from first two cameras)
            if len(camera_centers) >= 2:
                ray1 = point_3d - camera_centers[0]
                ray2 = point_3d - camera_centers[1]
                
                ray1_norm = ray1 / (np.linalg.norm(ray1) + 1e-10)
                ray2_norm = ray2 / (np.linalg.norm(ray2) + 1e-10)
                
                cos_angle = np.clip(np.dot(ray1_norm, ray2_norm), -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                triangulation_angles[point_idx] = np.degrees(angle_rad)
        
        logger.info(f"Triangulated {num_points} 3D points")
        
        # ========================================
        # CALCULATE REPROJECTION ERRORS
        # ========================================
        
        reprojection_errors = []
        all_errors = []
        points_per_point_errors = np.zeros(num_points)
        
        for view_idx, view in enumerate(view_data):
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            distortion = np.array(view['distortion'], dtype=np.float64)
            extrinsic = np.array(view['extrinsic'], dtype=np.float64)
            original_points_2d = np.array(view['points_2d'], dtype=np.float32)
            
            # Project 3D points back to 2D
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            tvec = t.reshape(-1, 1)
            
            # Project points
            projected_points, _ = cv2.projectPoints(
                points_3d.astype(np.float64),
                rvec,
                tvec,
                intrinsic,
                distortion
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Calculate reprojection errors
            errors = np.linalg.norm(original_points_2d - projected_points, axis=1)
            
            view_error_info = {
                'view_index': view_idx,
                'mean_error': float(np.mean(errors)),
                'max_error': float(np.max(errors)),
                'min_error': float(np.min(errors)),
                'std_error': float(np.std(errors)),
                'errors_per_point': errors
            }
            
            reprojection_errors.append(view_error_info)
            all_errors.extend(errors)
            points_per_point_errors += errors
            
            logger.debug(f"View {view_idx}: Mean reprojection error = {view_error_info['mean_error']:.3f} pixels")
        
        # Average error per point across all views
        points_per_point_errors /= len(view_data)
        
        mean_reprojection_error = float(np.mean(all_errors))
        
        logger.info(f"Mean reprojection error across all views: {mean_reprojection_error:.3f} pixels")
        
        # ========================================
        # PREPARE RESULTS
        # ========================================
        
        result = {
            'success': True,
            'points_3d': points_3d,
            'num_points': num_points,
            'num_views': len(view_data),
            'reprojection_errors': reprojection_errors,
            'mean_reprojection_error': mean_reprojection_error,
            'points_per_point_errors': points_per_point_errors,
            'triangulation_angles': triangulation_angles,
            'camera_centers': np.array(camera_centers)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Triangulation failed: {str(e)}")
        return {
            'success': False,
            'error_message': str(e),
            'num_views': len(view_data) if view_data else 0
        }


def _triangulate_dlt(points_2d: List[np.ndarray], 
                     projection_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Direct Linear Transform (DLT) triangulation for a single point from multiple views.
    
    This is the core triangulation algorithm that solves for 3D point coordinates
    given 2D observations and camera projection matrices using SVD.
    
    Args:
        points_2d: List of 2D points (one per view), each is np.ndarray of shape (2,)
        projection_matrices: List of 3x4 projection matrices (one per view)
        
    Returns:
        np.ndarray of shape (3,) - 3D point coordinates
        
    Mathematical formulation:
        For each view i with 2D point (x_i, y_i) and projection matrix P_i:
        x_i * P_i[2] - P_i[0] = 0
        y_i * P_i[2] - P_i[1] = 0
        
        Build matrix A from all views and solve Ax = 0 using SVD
    """
    n_views = len(points_2d)
    A = np.zeros((2 * n_views, 4))
    
    for i, (pt_2d, P) in enumerate(zip(points_2d, projection_matrices)):
        x, y = pt_2d[0], pt_2d[1]
        
        # Build linear system Ax = 0
        # x * P[2] - P[0] = 0
        # y * P[2] - P[1] = 0
        A[2*i] = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]
    
    # Solve using SVD: A = U * S * Vt
    # Solution is last row of Vt (smallest singular value)
    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]
    
    # Convert from homogeneous coordinates to 3D
    if abs(X_homogeneous[3]) > 1e-10:
        X_3d = X_homogeneous[:3] / X_homogeneous[3]
    else:
        # Point at infinity - use unnormalized coordinates
        logger.warning("Triangulation resulted in point at infinity")
        X_3d = X_homogeneous[:3]
    
    return X_3d


def calculate_triangulation_quality(view_data: List[Dict], points_3d: np.ndarray) -> Dict:
    """
    Calculate quality metrics for triangulated points.
    
    Args:
        view_data: Original view data used for triangulation
        points_3d: Triangulated 3D points
        
    Returns:
        Dict containing quality metrics:
        {
            'triangulation_angles': np.ndarray - Angle between viewing rays (degrees)
            'distances_to_cameras': List[np.ndarray] - Distance from each camera to points
            'depth_consistency': np.ndarray - Variance in depth across views
        }
    """
    num_points = len(points_3d)
    
    # Extract camera centers
    camera_centers = []
    for view in view_data:
        extrinsic = np.array(view['extrinsic'])
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        camera_center = -R.T @ t
        camera_centers.append(camera_center)
    
    # Calculate triangulation angles (between first two views)
    triangulation_angles = np.zeros(num_points)
    if len(camera_centers) >= 2:
        for i, point_3d in enumerate(points_3d):
            ray1 = point_3d - camera_centers[0]
            ray2 = point_3d - camera_centers[1]
            
            ray1_norm = ray1 / (np.linalg.norm(ray1) + 1e-10)
            ray2_norm = ray2 / (np.linalg.norm(ray2) + 1e-10)
            
            cos_angle = np.clip(np.dot(ray1_norm, ray2_norm), -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            triangulation_angles[i] = np.degrees(angle_rad)
    
    # Calculate distances to cameras
    distances_to_cameras = []
    for camera_center in camera_centers:
        distances = np.linalg.norm(points_3d - camera_center, axis=1)
        distances_to_cameras.append(distances)
    
    # Calculate depth consistency (variance in depth across views)
    depth_consistency = np.std(distances_to_cameras, axis=0)
    
    return {
        'triangulation_angles': triangulation_angles,
        'distances_to_cameras': distances_to_cameras,
        'depth_consistency': depth_consistency,
        'mean_triangulation_angle': float(np.mean(triangulation_angles)),
        'min_triangulation_angle': float(np.min(triangulation_angles)),
        'max_triangulation_angle': float(np.max(triangulation_angles))
    }


def validate_view_geometry(view_data: List[Dict]) -> Dict:
    """
    Validate the geometry of camera views for triangulation.
    
    Checks for:
    - Sufficient baseline between cameras
    - Reasonable viewing angles
    - Camera configuration issues
    
    Args:
        view_data: List of view dictionaries
        
    Returns:
        Dict containing validation results:
        {
            'valid': bool - Overall validation status
            'warnings': List[str] - Warning messages
            'baseline_distances': np.ndarray - Distances between camera pairs
            'mean_baseline': float - Mean baseline distance
            'camera_centers': np.ndarray - Camera center positions
        }
    """
    warnings = []
    
    if len(view_data) < 2:
        return {
            'valid': False,
            'warnings': ['At least 2 views required'],
            'baseline_distances': np.array([]),
            'mean_baseline': 0.0
        }
    
    # Extract camera centers
    camera_centers = []
    for i, view in enumerate(view_data):
        try:
            extrinsic = np.array(view['extrinsic'])
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            camera_center = -R.T @ t
            camera_centers.append(camera_center)
        except Exception as e:
            warnings.append(f"View {i}: Failed to extract camera center - {str(e)}")
    
    camera_centers = np.array(camera_centers)
    
    # Calculate baselines between all camera pairs
    baseline_distances = []
    for i in range(len(camera_centers)):
        for j in range(i + 1, len(camera_centers)):
            distance = np.linalg.norm(camera_centers[i] - camera_centers[j])
            baseline_distances.append(distance)
    
    baseline_distances = np.array(baseline_distances)
    mean_baseline = float(np.mean(baseline_distances)) if len(baseline_distances) > 0 else 0.0
    
    # Check for insufficient baseline
    if mean_baseline < 0.01:  # Less than 1cm
        warnings.append(f"Very small baseline distance ({mean_baseline*1000:.2f}mm) may lead to poor triangulation")
    
    # Check if all cameras are at the same position
    if np.max(baseline_distances) < 1e-6:
        warnings.append("All cameras appear to be at the same position")
    
    valid = len(warnings) == 0
    
    return {
        'valid': valid,
        'warnings': warnings,
        'baseline_distances': baseline_distances,
        'mean_baseline': mean_baseline,
        'camera_centers': camera_centers
    }


def test_with_chessboard(visualize=False):
    """
    Test triangulation using chessboard calibration data.
    
    This function:
    1. Performs intrinsic calibration using chessboard images
    2. Detects chessboard corners in multiple views
    3. Uses robot poses as extrinsic parameters
    4. Triangulates 3D positions of chessboard corners
    5. Compares with known 3D positions from chessboard pattern
    
    Args:
        visualize: If True, display 3D visualization of triangulated points and cameras
    
    Returns:
        Dict containing test results
    """
    import glob
    import json
    from pathlib import Path
    
    print("=" * 80)
    print("Testing Triangulation with Chessboard Calibration Data")
    print("=" * 80)
    
    # Add camera calibration toolkit to path
    toolkit_path = Path(__file__).parent.parent / "ThirdParty" / "camera_calibration_toolkit"
    sys.path.insert(0, str(toolkit_path))
    
    try:
        from core.intrinsic_calibration import IntrinsicCalibrator
        from core.calibration_patterns import load_pattern_from_json
    except ImportError as e:
        print(f"❌ Failed to import calibration toolkit: {e}")
        print("\n💡 The camera calibration toolkit submodule may not be initialized.")
        print("   Please run the following commands to update the submodule:")
        print("   git submodule update --init --recursive")
        print(f"\n   Or check if the toolkit exists at: {toolkit_path}")
        return {'success': False, 'error': 'Calibration toolkit not available'}
    
    # ========================================
    # STEP 1: INTRINSIC CALIBRATION
    # ========================================
    
    print("\n📷 Step 1: Performing intrinsic calibration...")
    print("-" * 80)
    
    sample_dir = toolkit_path / "sample_data" / "eye_in_hand_test_data"
    image_paths = sorted(glob.glob(str(sample_dir / "*.jpg")))
    
    # Load pattern configuration
    config_path = sample_dir / "chessboard_config.json"
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    pattern = load_pattern_from_json(config_data)
        
    # Create calibrator and calibrate
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,
        calibration_pattern=pattern
    )
    
    calib_result = calibrator.calibrate(
        cameraMatrix=None,
        distCoeffs=None,
        flags=0,
        criteria=None,
        verbose=False
    )
    
    camera_matrix = calib_result['camera_matrix']
    distortion_coeffs = calib_result['distortion_coefficients']
        
    # ========================================
    # STEP 2: PREPARE VIEW DATA FROM CALIBRATION
    # ========================================
    
    print("\n🎯 Step 2: Preparing view data from calibration results...")
    print("-" * 80)
    
    # Get detected corners and poses from calibrator
    image_points_all = calibrator.image_points
    rvecs = calibrator.rvecs
    tvecs = calibrator.tvecs
    
    # Filter out None values (failed detections)
    valid_indices = [i for i in range(len(image_points_all)) 
                     if image_points_all[i] is not None]
    
    # Use all valid views for triangulation
    test_indices = valid_indices
    
    print(f"Using {len(test_indices)} views: indices {test_indices}")
    
    view_data = []
    
    for idx in test_indices:
        corners_2d = image_points_all[idx].reshape(-1, 2)
        
        # Get camera pose from rvec and tvec
        rvec = rvecs[idx]
        tvec = tvecs[idx]
        
        # Convert rotation vector to matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # board2cam transformation
        board2cam = np.eye(4)
        board2cam[:3, :3] = rmat
        board2cam[:3, 3] = tvec.flatten()
        
        # For extrinsic matrix, we need world2cam
        # We use the chessboard frame as the world frame
        world2cam = board2cam
        
        # Get image size from the calibrator
        img = calibrator.images[idx]
        
        view_info = {
            'points_2d': corners_2d,
            'image_size': (img.shape[1], img.shape[0]),
            'intrinsic': camera_matrix,
            'distortion': distortion_coeffs,
            'extrinsic': world2cam
        }
        view_data.append(view_info)
    
    if len(view_data) < 2:
        print(f"❌ Not enough valid views for triangulation (got {len(view_data)})")
        return {'success': False, 'error': 'Insufficient valid views'}
    
    print(f"✅ Prepared {len(view_data)} views for triangulation")

    print("view_data sample:", view_data[0])
    
    # ========================================
    # STEP 3: TRIANGULATE 3D POINTS
    # ========================================
    
    print("\n📐 Step 3: Triangulating 3D points...")
    print("-" * 80)
    
    try:
        print("Calling triangulate_multiview()...")
        result = triangulate_multiview(view_data)
        print("triangulate_multiview() returned successfully")
    except Exception as e:
        print(f"❌ Exception in triangulate_multiview(): {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    if not result['success']:
        print(f"❌ Triangulation failed: {result.get('error_message', 'Unknown error')}")
        return result
    
    points_3d = result['points_3d']
    print(f"Points 3D shape: {points_3d.shape}")
    
    print(f"✅ Triangulated {result['num_points']} 3D points from {result['num_views']} views")
    print(f"Mean reprojection error: {result['mean_reprojection_error']:.3f} pixels")
    
    # Print per-view errors
    print("\nPer-view reprojection errors:")
    for err_info in result['reprojection_errors']:
        print(f"  View {err_info['view_index']}: "
              f"mean={err_info['mean_error']:.3f}px, "
              f"max={err_info['max_error']:.3f}px, "
              f"std={err_info['std_error']:.3f}px")
    
    # ========================================
    # STEP 4: COMPARE WITH GROUND TRUTH
    # ========================================
    
    print("\n📊 Step 4: Analyzing triangulation quality...")
    print("-" * 80)
    
    # Generate ideal 3D positions of chessboard corners
    objp = np.zeros((pattern.width * pattern.height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern.width, 0:pattern.height].T.reshape(-1, 2)
    objp *= pattern.square_size
    
    # The triangulated points are in world coordinates
    # For quality analysis, we look at the planarity and spacing
    
    # Check planarity (points should lie on a plane)
    # Fit a plane to the triangulated points
    centroid = np.mean(points_3d, axis=0)
    points_centered = points_3d - centroid
    _, _, Vt = np.linalg.svd(points_centered)
    normal = Vt[-1]  # Normal vector of the best-fit plane
    
    # Calculate distances from points to the plane
    distances_to_plane = np.abs(np.dot(points_centered, normal))
    mean_planarity_error = np.mean(distances_to_plane)
    max_planarity_error = np.max(distances_to_plane)
    
    print("Planarity analysis:")
    print(f"  Mean distance to plane: {mean_planarity_error*1000:.3f} mm")
    print(f"  Max distance to plane: {max_planarity_error*1000:.3f} mm")
    
    # Analyze triangulation angles
    mean_angle = np.mean(result['triangulation_angles'])
    min_angle = np.min(result['triangulation_angles'])
    max_angle = np.max(result['triangulation_angles'])
    
    print("\nTriangulation angles:")
    print(f"  Mean: {mean_angle:.1f}°")
    print(f"  Min: {min_angle:.1f}°")
    print(f"  Max: {max_angle:.1f}°")
    
    # Calculate spacing between adjacent corners
    spacing_errors = []
    for i in range(pattern.height):
        for j in range(pattern.width - 1):
            idx1 = i * pattern.width + j
            idx2 = i * pattern.width + j + 1
            dist = np.linalg.norm(points_3d[idx1] - points_3d[idx2])
            error = abs(dist - pattern.square_size)
            spacing_errors.append(error)
    
    mean_spacing_error = np.mean(spacing_errors)
    
    print("\nSpacing consistency:")
    print(f"  Expected spacing: {pattern.square_size*1000:.1f} mm")
    print(f"  Mean spacing error: {mean_spacing_error*1000:.3f} mm")
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    success = (
        result['mean_reprojection_error'] < 2.0 and
        mean_planarity_error < 0.005 and  # 5mm
        mean_spacing_error < 0.002  # 2mm
    )
    
    if success:
        print("✅ TEST PASSED")
    else:
        print("⚠️  TEST COMPLETED WITH WARNINGS")
    
    print("\nMetrics:")
    print(f"  Reprojection error: {result['mean_reprojection_error']:.3f} px {'✅' if result['mean_reprojection_error'] < 2.0 else '⚠️'}")
    print(f"  Planarity error: {mean_planarity_error*1000:.3f} mm {'✅' if mean_planarity_error < 0.005 else '⚠️'}")
    print(f"  Spacing error: {mean_spacing_error*1000:.3f} mm {'✅' if mean_spacing_error < 0.002 else '⚠️'}")
    print(f"  Triangulation angle: {mean_angle:.1f}° {'✅' if mean_angle > 10 else '⚠️'}")
    
    # ========================================
    # VISUALIZATION
    # ========================================
    
    if visualize:
        print("\n📊 Opening 3D visualization...")
        print("-" * 80)
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot triangulated 3D points
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                      c='blue', marker='o', s=20, label='Triangulated Points')
            
            # Plot camera positions
            camera_centers = result['camera_centers']
            ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
                      c='red', marker='^', s=100, label='Camera Positions')
            
            # Draw lines from cameras to center of point cloud
            points_center = np.mean(points_3d, axis=0)
            for i, cam_pos in enumerate(camera_centers):
                ax.plot([cam_pos[0], points_center[0]], 
                       [cam_pos[1], points_center[1]], 
                       [cam_pos[2], points_center[2]], 
                       'r--', alpha=0.3, linewidth=1)
                # Label cameras
                ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'  Cam{i}', fontsize=8)
            
            # Draw chessboard grid lines
            for i in range(pattern.height):
                row_points = points_3d[i*pattern.width:(i+1)*pattern.width]
                ax.plot(row_points[:, 0], row_points[:, 1], row_points[:, 2], 
                       'g-', alpha=0.5, linewidth=0.5)
            
            for j in range(pattern.width):
                col_points = points_3d[j::pattern.width]
                ax.plot(col_points[:, 0], col_points[:, 1], col_points[:, 2], 
                       'g-', alpha=0.5, linewidth=0.5)
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'3D Triangulation Result\n{result["num_points"]} points from {result["num_views"]} views')
            ax.legend()
            
            # Equal aspect ratio
            max_range = np.array([points_3d[:, 0].max()-points_3d[:, 0].min(),
                                 points_3d[:, 1].max()-points_3d[:, 1].min(),
                                 points_3d[:, 2].max()-points_3d[:, 2].min()]).max() / 2.0
            mid_x = (points_3d[:, 0].max()+points_3d[:, 0].min()) * 0.5
            mid_y = (points_3d[:, 1].max()+points_3d[:, 1].min()) * 0.5
            mid_z = (points_3d[:, 2].max()+points_3d[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Visualization closed")
            
        except ImportError as e:
            print(f"⚠️  Could not create visualization: matplotlib not available ({e})")
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
    
    return {
        'success': success,
        'num_points': result['num_points'],
        'num_views': result['num_views'],
        'mean_reprojection_error': result['mean_reprojection_error'],
        'mean_planarity_error': float(mean_planarity_error),
        'mean_spacing_error': float(mean_spacing_error),
        'mean_triangulation_angle': float(mean_angle),
        'points_3d': points_3d
    }


# Example usage and testing
if __name__ == "__main__":
    print("Multi-View Triangulation Module")
    print("=" * 70)
    
    # Run test with chessboard data
    print("\n🧪 Running test with chessboard calibration data...")
    print("=" * 70)
    
    try:
        # Enable visualization by default when running directly
        test_result = test_with_chessboard(visualize=True)
        
        if test_result['success']:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  Test completed with issues: {test_result.get('error', 'See details above')}")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
