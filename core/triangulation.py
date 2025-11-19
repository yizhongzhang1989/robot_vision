"""
Multi-View Triangulation Module

This module provides triangulation functionality for estimating 3D coordinates
from multiple 2D views with camera parameters.

Author: Yizhong Zhang
Date: November 2025
"""

import numpy as np
import cv2
from typing import List, Dict


def triangulate_multiview(view_data: List[Dict]) -> Dict:
    """
    Triangulate 3D points from multiple 2D views with camera parameters.
    
    This function takes 2D point observations from multiple camera views along with
    their intrinsic and extrinsic parameters, and estimates the corresponding 3D
    coordinates using Direct Linear Transform (DLT) triangulation.
    
    Args:
        view_data: List of view dictionaries. Each view should contain:
            - 'points_2d': np.ndarray of shape (N, 2) - 2D pixel coordinates (x, y)
                          in OpenCV convention where top-left pixel center is at (0, 0)
            - 'image_size': tuple (width, height) - Image dimensions in pixels
            - 'intrinsic': np.ndarray of shape (3, 3) - Camera intrinsic matrix (K)
                          [[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]]
            - 'distortion': (optional) np.ndarray - Distortion coefficients [k1, k2, p1, p2, k3, ...]
                           If omitted, points are assumed to be pre-undistorted
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
            'reprojection_errors': List[np.ndarray] - Per-point reprojection errors for each view
                List of length num_views, where each element is an np.ndarray of shape (N,)
                containing the reprojection error in pixels for each of the N points in that view
            'error_message': str - Error message if success is False (only present when success=False)
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
            reprojection_errors = result['reprojection_errors']
            mean_error = np.mean([np.mean(errors) for errors in reprojection_errors])
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
        # Note: 'distortion' is optional - if missing, points are assumed to be pre-undistorted
        required_fields = ['points_2d', 'image_size', 'intrinsic', 'extrinsic']
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
        
        # ========================================
        # UNDISTORT POINTS
        # ========================================
        
        undistorted_points = []
        
        for i, view in enumerate(view_data):
            points_2d = np.array(view['points_2d'], dtype=np.float32)
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            
            # Check if distortion is provided
            if 'distortion' in view:
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
            else:
                # Points are already undistorted
                undistorted_points.append(points_2d)
        
        # ========================================
        # PREPARE PROJECTION MATRICES
        # ========================================
        
        projection_matrices = []
        camera_centers = []
        
        for i, view in enumerate(view_data):
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            extrinsic = np.array(view['extrinsic'], dtype=np.float64)
            
            try:
                # Extract rotation and translation from extrinsic matrix
                R = extrinsic[:3, :3]
                t = extrinsic[:3, 3]
                
                # Create [R|t] matrix
                RT = np.hstack([R, t.reshape(-1, 1)])
                
                # Projection matrix P = K * [R|t]
                P = intrinsic @ RT
                projection_matrices.append(P)
            except Exception:
                raise
            
            # Calculate camera center in world coordinates
            # Camera center: C = -R^T * t
            camera_center = -R.T @ t
            camera_centers.append(camera_center)
        
        # ========================================
        # TRIANGULATE POINTS
        # ========================================
        
        points_3d = np.zeros((num_points, 3))
        
        for point_idx in range(num_points):
            # Collect 2D observations for this point from all views
            observations_2d = [undistorted_points[view_idx][point_idx] 
                              for view_idx in range(len(view_data))]
            
            # Triangulate using DLT
            point_3d = _triangulate_dlt(observations_2d, projection_matrices)
            points_3d[point_idx] = point_3d
        
        # ========================================
        # CALCULATE REPROJECTION ERRORS
        # ========================================
        
        # reprojection_errors: List of per-point errors for each view
        # Shape: List[np.ndarray] where each array has shape (num_points,)
        reprojection_errors = []
        
        for view_idx, view in enumerate(view_data):
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            extrinsic = np.array(view['extrinsic'], dtype=np.float64)
            original_points_2d = np.array(view['points_2d'], dtype=np.float32)
            
            # Check if distortion is provided
            if 'distortion' in view:
                distortion = np.array(view['distortion'], dtype=np.float64)
            else:
                # No distortion - use zero distortion coefficients
                distortion = np.zeros(5, dtype=np.float64)
            
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
            
            # Calculate reprojection errors for each point
            errors = np.linalg.norm(original_points_2d - projected_points, axis=1)
            reprojection_errors.append(errors)
        
        # ========================================
        # PREPARE RESULTS
        # ========================================
        
        result = {
            'success': True,
            'points_3d': points_3d,
            'reprojection_errors': reprojection_errors
        }
        
        return result
        
    except Exception as e:
        # Sanitize error message for Windows console encoding issues
        try:
            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        except Exception:
            error_msg = "Unknown error (encoding issue)"
        
        return {
            'success': False,
            'error_message': error_msg,
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
        X_3d = X_homogeneous[:3]
    
    return X_3d


def triangulate_view_plane(view_data: Dict, plane_point: np.ndarray, plane_normal: np.ndarray) -> Dict:
    """
    Triangulate 3D points by intersecting camera rays with a plane.
    
    This function takes 2D point observations from a single camera view and finds
    their 3D positions on a known plane by intersecting the camera rays with that plane.
    
    Args:
        view_data: Dictionary containing single view data:
            - 'points_2d': np.ndarray of shape (N, 2) - 2D pixel coordinates (x, y)
            - 'image_size': tuple (width, height) - Image dimensions in pixels
            - 'intrinsic': np.ndarray of shape (3, 3) - Camera intrinsic matrix
            - 'distortion': (optional) np.ndarray - Distortion coefficients
            - 'extrinsic': np.ndarray of shape (4, 4) - Camera extrinsic matrix (world to camera)
        plane_point: np.ndarray of shape (3,) - A point on the plane in world coordinates
        plane_normal: np.ndarray of shape (3,) - Normal vector of the plane (will be normalized)
    
    Returns:
        Dict containing triangulation results:
        {
            'success': bool - Whether triangulation succeeded
            'points_3d': np.ndarray of shape (N, 3) - 3D points on the plane in world coordinates
            'distances': np.ndarray of shape (N,) - Distance from camera center to each 3D point
            'error_message': str - Error message if success is False (only present when success=False)
        }
    
    Example usage:
        view_data = {
            'points_2d': np.array([[100, 200], [150, 180]]),
            'image_size': (1920, 1080),
            'intrinsic': intrinsic_matrix,
            'distortion': distortion_coeffs,
            'extrinsic': extrinsic_matrix
        }
        
        plane_point = np.array([0.0, 0.0, 0.0])  # Origin on the plane
        plane_normal = np.array([0.0, 0.0, 1.0])  # XY plane (Z=0)
        
        result = triangulate_view_plane(view_data, plane_point, plane_normal)
        
        if result['success']:
            points_3d = result['points_3d']
            print(f"Triangulated {len(points_3d)} points on the plane")
    """
    try:
        # ========================================
        # INPUT VALIDATION
        # ========================================
        
        required_fields = ['points_2d', 'image_size', 'intrinsic', 'extrinsic']
        for field in required_fields:
            if field not in view_data:
                return {
                    'success': False,
                    'error_message': f"Missing required field '{field}' in view_data"
                }
        
        points_2d = np.array(view_data['points_2d'], dtype=np.float32)
        num_points = len(points_2d)
        
        if num_points == 0:
            return {
                'success': False,
                'error_message': 'No points to triangulate'
            }
        
        plane_point = np.array(plane_point, dtype=np.float64)
        plane_normal = np.array(plane_normal, dtype=np.float64)
        
        if plane_point.shape != (3,):
            return {
                'success': False,
                'error_message': f'plane_point must have shape (3,), got {plane_point.shape}'
            }
        
        if plane_normal.shape != (3,):
            return {
                'success': False,
                'error_message': f'plane_normal must have shape (3,), got {plane_normal.shape}'
            }
        
        # Normalize plane normal
        plane_normal_norm = np.linalg.norm(plane_normal)
        if plane_normal_norm < 1e-10:
            return {
                'success': False,
                'error_message': 'plane_normal has zero length'
            }
        plane_normal = plane_normal / plane_normal_norm
        
        # ========================================
        # UNDISTORT POINTS
        # ========================================
        
        intrinsic = np.array(view_data['intrinsic'], dtype=np.float64)
        
        if 'distortion' in view_data:
            distortion = np.array(view_data['distortion'], dtype=np.float64)
            points_reshaped = points_2d.reshape(-1, 1, 2)
            
            # Undistort to normalized camera coordinates
            undistorted_normalized = cv2.undistortPoints(
                points_reshaped,
                intrinsic,
                distortion,
                P=None  # Return normalized coordinates
            )
            undistorted_normalized = undistorted_normalized.reshape(-1, 2)
        else:
            # Convert pixel coordinates to normalized camera coordinates
            # [x_norm, y_norm] = K^-1 * [x_pixel, y_pixel, 1]
            intrinsic_inv = np.linalg.inv(intrinsic)
            points_homogeneous = np.hstack([points_2d, np.ones((num_points, 1))])
            normalized_coords = (intrinsic_inv @ points_homogeneous.T).T
            undistorted_normalized = normalized_coords[:, :2]
        
        # ========================================
        # COMPUTE CAMERA CENTER AND ORIENTATION
        # ========================================
        
        extrinsic = np.array(view_data['extrinsic'], dtype=np.float64)
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # Camera center in world coordinates: C = -R^T * t
        camera_center = -R.T @ t
        
        # ========================================
        # RAY-PLANE INTERSECTION
        # ========================================
        
        points_3d = np.zeros((num_points, 3))
        distances = np.zeros(num_points)
        
        for i in range(num_points):
            # Ray direction in camera coordinates (normalized)
            ray_camera = np.array([undistorted_normalized[i, 0], 
                                   undistorted_normalized[i, 1], 
                                   1.0])
            
            # Transform ray to world coordinates
            ray_world = R.T @ ray_camera
            ray_world = ray_world / np.linalg.norm(ray_world)
            
            # Ray-plane intersection
            # Plane equation: (P - plane_point) · plane_normal = 0
            # Ray equation: P = camera_center + d * ray_world
            # Solve for d: (camera_center + d * ray_world - plane_point) · plane_normal = 0
            
            denominator = np.dot(ray_world, plane_normal)
            
            if abs(denominator) < 1e-10:
                # Ray is parallel to plane - leave as zero (invalid)
                continue
            
            numerator = np.dot(plane_point - camera_center, plane_normal)
            d = numerator / denominator
            
            if d < 0:
                # Intersection point is behind the camera - leave as zero (invalid)
                continue
            
            # 3D point on plane
            point_3d = camera_center + d * ray_world
            points_3d[i] = point_3d
            distances[i] = d
        
        # ========================================
        # PREPARE RESULTS
        # ========================================
        
        result = {
            'success': True,
            'points_3d': points_3d,
            'distances': distances
        }
        
        return result
        
    except Exception as e:
        # Sanitize error message for encoding issues
        try:
            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        except Exception:
            error_msg = "Unknown error (encoding issue)"
        
        return {
            'success': False,
            'error_message': error_msg
        }
