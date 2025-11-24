"""
Multi-View Triangulation and Fitting Module

This module provides functionality for:
1. Triangulation: Estimating 3D coordinates from multiple 2D views
2. Fitting: Estimating rigid transformations from known 3D points and multi-view observations

Author: Yizhong Zhang
Date: November 2025
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from scipy.optimize import least_squares


def triangulate_multiview(view_data: List[Dict]) -> Dict:
    """
    Triangulate 3D points from multiple 2D views with camera parameters.
    
    This function takes 2D point observations from multiple camera views along with
    their intrinsic and extrinsic parameters, and estimates the corresponding 3D
    coordinates using Direct Linear Transform (DLT) triangulation.
    
    Args:
        view_data: List of view dictionaries. Each view should contain:
            - 'points_2d': List or np.ndarray of length N - 2D pixel coordinates
                          Each element can be:
                          - np.ndarray of shape (2,) with (x, y) coordinates
                          - None to indicate the point is not detected in this view
                          All views must have the same length N (same number of points)
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
                'points_2d': [np.array([100.5, 200.3]), None, np.array([150.2, 180.7])],
                'image_size': (1920, 1080),
                'intrinsic': np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]]),
                'distortion': np.array([0.1, -0.05, 0, 0, 0]),
                'extrinsic': np.eye(4)  # World to camera transformation
            }
            Note: Second point is None, indicating it's not detected in this view
    
    Returns:
        Dict containing triangulation results:
        {
            'success': bool - Whether triangulation succeeded
            'points_3d': List[np.ndarray or None] - Triangulated 3D points in world coordinates
                List of length N, where each element is either:
                - np.ndarray of shape (3,) for successfully triangulated points
                - None for points with insufficient valid views (< 2 views)
            'reprojection_errors': List[List[float or None]] - Per-point reprojection errors
                List of length num_views, where each element is a list of length N
                containing reprojection error in pixels or None for missing detections
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
            points_2d_list = view['points_2d']
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            
            # Separate valid points from None values
            valid_indices = [idx for idx, pt in enumerate(points_2d_list) if pt is not None]
            valid_points = [pt for pt in points_2d_list if pt is not None]
            
            if len(valid_points) > 0:
                valid_points_array = np.array(valid_points, dtype=np.float32)
                
                # Check if distortion is provided
                if 'distortion' in view:
                    distortion = np.array(view['distortion'], dtype=np.float64)
                    
                    # Reshape for cv2.undistortPoints
                    points_reshaped = valid_points_array.reshape(-1, 1, 2)
                    
                    # Undistort points and normalize to camera coordinates
                    # Then project back to pixel coordinates using intrinsic matrix
                    undistorted = cv2.undistortPoints(
                        points_reshaped,
                        intrinsic,
                        distortion,
                        P=intrinsic
                    )
                    
                    undistorted_valid = undistorted.reshape(-1, 2)
                else:
                    # Points are already undistorted
                    undistorted_valid = valid_points_array
                
                # Reconstruct full list with None for missing points
                undistorted_full = [None] * num_points
                for valid_idx, undist_pt in zip(valid_indices, undistorted_valid):
                    undistorted_full[valid_idx] = undist_pt
                
                undistorted_points.append(undistorted_full)
            else:
                # All points are None for this view
                undistorted_points.append([None] * num_points)
        
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
        
        points_3d = []
        
        for point_idx in range(num_points):
            # Collect valid 2D observations for this point from all views
            valid_observations = []
            valid_projections = []
            
            for view_idx in range(len(view_data)):
                obs = undistorted_points[view_idx][point_idx]
                if obs is not None:
                    valid_observations.append(obs)
                    valid_projections.append(projection_matrices[view_idx])
            
            # Need at least 2 valid views to triangulate
            if len(valid_observations) >= 2:
                # Triangulate using DLT
                point_3d = _triangulate_dlt(valid_observations, valid_projections)
                points_3d.append(point_3d)
            else:
                # Not enough valid views for this point
                points_3d.append(None)
        
        # ========================================
        # CALCULATE REPROJECTION ERRORS
        # ========================================
        
        # reprojection_errors: List of per-point errors for each view
        # Shape: List[List] where each list has length num_points
        # Elements are either float (error in pixels) or None (missing detection or invalid 3D point)
        reprojection_errors = []
        
        for view_idx, view in enumerate(view_data):
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            extrinsic = np.array(view['extrinsic'], dtype=np.float64)
            original_points_2d_list = view['points_2d']
            
            # Check if distortion is provided
            if 'distortion' in view:
                distortion = np.array(view['distortion'], dtype=np.float64)
            else:
                # No distortion - use zero distortion coefficients
                distortion = np.zeros(5, dtype=np.float64)
            
            # Prepare for projection
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            tvec = t.reshape(-1, 1)
            
            # Calculate error for each point
            view_errors = []
            for point_idx in range(num_points):
                original_2d = original_points_2d_list[point_idx]
                point_3d = points_3d[point_idx]
                
                # Skip if either 2D or 3D point is None
                if original_2d is None or point_3d is None:
                    view_errors.append(None)
                    continue
                
                # Project 3D point back to 2D
                projected_point, _ = cv2.projectPoints(
                    np.array([point_3d], dtype=np.float64),
                    rvec,
                    tvec,
                    intrinsic,
                    distortion
                )
                projected_point = projected_point.reshape(2)
                
                # Calculate reprojection error
                original_2d_array = np.array(original_2d, dtype=np.float32)
                error = np.linalg.norm(original_2d_array - projected_point)
                view_errors.append(float(error))
            
            reprojection_errors.append(view_errors)
        
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


def fitting_multiview(view_data: List[Dict], target_point3d_local: List[np.ndarray]) -> Dict:
    """
    Estimate local->world rigid transform from known 3D points and multi-view observations.
    
    This function takes 2D observations of points with known 3D local coordinates from
    multiple camera views and estimates the rigid transformation (rotation and translation)
    that maps from the local coordinate frame to the world coordinate frame.
    
    Args:
        view_data: List of view dictionaries (same format as triangulate_multiview):
            - 'points_2d': List of length N - 2D pixel coordinates or None
            - 'image_size': tuple (width, height) - Image dimensions
            - 'intrinsic': np.ndarray of shape (3, 3) - Camera intrinsic matrix
            - 'distortion': (optional) np.ndarray - Distortion coefficients
            - 'extrinsic': np.ndarray of shape (4, 4) - World to camera transformation
        target_point3d_local: List of N np.ndarray of shape (3,) - 3D points in local coordinates
    
    Returns:
        Dict containing fitting results:
        {
            'success': bool - Whether fitting succeeded
            'local2world': np.ndarray of shape (4, 4) - Transformation matrix from local to world
            'points_3d': List[np.ndarray] - 3D points in world coordinates (N points of shape (3,))
            'reprojection_errors': List[List[float or None]] - Per-point reprojection errors
            'error_message': str - Error message if success is False (only present when success=False)
        }
    
    Example usage:
        # Known 3D points in local coordinate frame (e.g., chessboard corners)
        target_points_local = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.02, 0.0, 0.0]),
            np.array([0.0, 0.02, 0.0]),
            # ... more points
        ]
        
        # Multi-view 2D observations
        view_data = [
            {
                'points_2d': [np.array([100, 200]), np.array([150, 200]), ...],
                'image_size': (1920, 1080),
                'intrinsic': intrinsic_matrix,
                'distortion': distortion_coeffs,
                'extrinsic': extrinsic_matrix
            },
            # ... more views
        ]
        
        result = fitting_multiview(view_data, target_points_local)
        
        if result['success']:
            local2world = result['local2world']
            print(f"Transformation matrix:\\n{local2world}")
    """
    try:
        # --- input validation
        if not view_data or len(view_data) < 1:
            return {'success': False, 'error_message': 'At least 1 view required', 'num_views': len(view_data) if view_data else 0}
        N = len(target_point3d_local)
        for i, v in enumerate(view_data):
            if 'points_2d' not in v or 'intrinsic' not in v or 'extrinsic' not in v or 'image_size' not in v:
                return {'success': False, 'error_message': f'View {i} missing required fields', 'num_views': len(view_data)}
            if len(v['points_2d']) != N:
                return {'success': False, 'error_message': f'View {i} has {len(v["points_2d"])} points, expected {N}', 'num_views': len(view_data)}

        # convert target points to array
        model_pts = np.array([np.asarray(p, dtype=np.float64).reshape(3,) for p in target_point3d_local], dtype=np.float64)

        # --- Prepare per-view camera parameters (no undistortion preprocessing)
        view_K = []
        view_R = []
        view_t = []
        view_dist = []
        view_points_2d = []  # Store original distorted points
        
        for v in view_data:
            K = np.array(v['intrinsic'], dtype=np.float64)
            extrinsic = np.array(v['extrinsic'], dtype=np.float64)
            Rm = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            dist = np.array(v['distortion'], dtype=np.float64) if 'distortion' in v else np.zeros(5, dtype=np.float64)
            view_K.append(K)
            view_R.append(Rm)
            view_t.append(t)
            view_dist.append(dist)
            view_points_2d.append(v['points_2d'])

        # --- prepare "rays" in world coordinates for each observation
        # rays_per_obs: list (view) of lists (per point) of either None or (O_world, v_world)
        rays_per_view = []
        for vidx in range(len(view_data)):
            K = view_K[vidx]
            dist = view_dist[vidx]
            R_wc = view_R[vidx]  # world->camera
            t_wc = view_t[vidx]
            O_world = -R_wc.T @ t_wc
            view_rays = []
            
            for pi in range(N):
                p = view_points_2d[vidx][pi]
                if p is None:
                    view_rays.append(None)
                else:
                    # Convert distorted pixel to normalized camera coordinates using cv2.undistortPoints
                    p_array = np.array([[p]], dtype=np.float32)
                    normalized = cv2.undistortPoints(p_array, K, dist, P=None)
                    normalized = normalized.reshape(2,)
                    
                    # Create ray in camera coordinates
                    d_cam = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
                    d_cam = d_cam / np.linalg.norm(d_cam)
                    
                    # Transform to world coordinates
                    v_world = R_wc.T @ d_cam
                    v_world = v_world / np.linalg.norm(v_world)
                    view_rays.append((O_world.copy(), v_world.copy()))
            rays_per_view.append(view_rays)

        # --- TRY to triangulate model points that have >=2 observations (to get world points)
        # For triangulation, we need undistorted normalized coordinates
        triangulated_world = [None] * N
        for pi in range(N):
            # Collect undistorted normalized points for this point across views
            pts_normalized = []
            Ks = []
            Rs = []
            ts = []
            
            for vidx in range(len(view_data)):
                p = view_points_2d[vidx][pi]
                if p is not None:
                    # Undistort to normalized coordinates
                    p_array = np.array([[p]], dtype=np.float32)
                    normalized = cv2.undistortPoints(p_array, view_K[vidx], view_dist[vidx], P=None)
                    pts_normalized.append(normalized.reshape(2,))
                    Ks.append(view_K[vidx])
                    Rs.append(view_R[vidx])
                    ts.append(view_t[vidx])
            
            if len(pts_normalized) >= 2:
                try:
                    # Build projection matrices from undistorted normalized coords
                    # P = K @ [R|t], but for normalized coords we use identity K
                    proj_mats = []
                    for R, t in zip(Rs, ts):
                        RT = np.hstack([R, t.reshape(3, 1)])
                        proj_mats.append(RT)  # No K needed for normalized coords
                    
                    Xw = _triangulate_dlt_local(pts_normalized, proj_mats)
                    triangulated_world[pi] = Xw
                except Exception:
                    triangulated_world[pi] = None

        # --- If we have >=3 triangulated correspondences, use Umeyama for initial R,t
        tri_indices = [i for i, x in enumerate(triangulated_world) if x is not None]
        if len(tri_indices) >= 3:
            A = model_pts[tri_indices, :]
            B = np.vstack([triangulated_world[i] for i in tri_indices])
            _, R_init, t_init = _umeyama(A, B, with_scale=False)
            rotvec_init, _ = cv2.Rodrigues(R_init)
            rotvec_init = rotvec_init.flatten()
        else:
            # Fallback initial guess:
            # R = I, t shift model centroid to average of per-observation closest points.
            R_init = np.eye(3)
            rotvec_init = np.zeros(3, dtype=np.float64)
            centroid_model = model_pts.mean(axis=0) if len(model_pts) > 0 else np.zeros(3)
            closest_points = []
            for vidx in range(len(view_data)):
                for pi in range(N):
                    ray = rays_per_view[vidx][pi]
                    if ray is None:
                        continue
                    O, v = ray
                    # project centroid_model (in candidate world as if R=I,t=0) onto ray
                    # choose point on ray closest to centroid_model
                    w = centroid_model - O
                    s = np.dot(w, v)
                    p_closest = O + s * v
                    closest_points.append(p_closest)
            if len(closest_points) > 0:
                mean_p = np.mean(np.vstack(closest_points), axis=0)
                t_init = mean_p - (R_init @ centroid_model)
            else:
                # no observations at all? shouldn't happen due to validation, but be safe
                t_init = np.zeros(3, dtype=np.float64)

        x0 = np.hstack([rotvec_init.reshape(3,), t_init.reshape(3,)])

        # --- Define residuals for point-to-ray minimization ---
        obs_list = []
        for vidx in range(len(view_data)):
            for pi in range(N):
                ray = rays_per_view[vidx][pi]
                if ray is not None:
                    O, v = ray
                    obs_list.append((pi, O, v))

        def residuals_point_to_ray(x):
            rvec = x[0:3].reshape(3,)
            tvec = x[3:6].reshape(3,)
            Rm, _ = cv2.Rodrigues(rvec)
            res = []
            # For each observation (pi, O, v), compute perpendicular vector
            for (pi, O, v) in obs_list:
                X = model_pts[pi].reshape(3,)
                Xw = Rm @ X + tvec
                diff = Xw - O
                # perpendicular component
                perp = diff - v * (v.dot(diff))
                # append 3 components (optimizer will handle redundancy)
                res.extend(perp.tolist())
            return np.array(res, dtype=np.float64)

        # run coarse optimization: point-to-ray
        if len(obs_list) == 0:
            return {'success': False, 'error_message': 'No observations to fit', 'num_views': len(view_data)}

        # Use a robust loss and small tolerance
        res_coarse = least_squares(residuals_point_to_ray, x0, method='lm', max_nfev=200, verbose=0)
        x_coarse = res_coarse.x
        rvec_coarse = x_coarse[0:3]
        R_coarse, _ = cv2.Rodrigues(rvec_coarse)

        # --- Refinement: minimize pixel reprojection error ---
        # Build list of observed (view,point) to compute residuals in pixel space
        reproj_obs = []
        for vidx in range(len(view_data)):
            K = view_K[vidx]
            dist = view_dist[vidx]
            R_wc = view_R[vidx]
            t_wc = view_t[vidx]
            # We'll use cv2.projectPoints which expects rvec,tvec that map model->camera.
            # Note: we are solving local->world: X_world = R_local2world * X_local + t
            # Camera extrinsic is world->camera: X_cam = R_wc * X_world + t_wc
            # So chain: X_cam = R_wc * (R_local2world * X_local + t) + t_wc
            # => X_cam = (R_wc @ R_local2world) * X_local + (R_wc @ t + t_wc)
            for pi in range(N):
                orig_uv = view_data[vidx]['points_2d'][pi]
                if orig_uv is not None:
                    reproj_obs.append((vidx, pi))
        # prepare a mapping for reprojection residuals
        def residuals_reprojection(x):
            rvec_local = x[0:3].reshape(3,)
            t_local = x[3:6].reshape(3,)
            R_local, _ = cv2.Rodrigues(rvec_local)
            res = []
            for (vidx, pi) in reproj_obs:
                K = view_K[vidx]
                dist = view_dist[vidx] if view_dist[vidx] is not None else np.zeros(5, dtype=np.float64)
                R_wc = view_R[vidx]
                t_wc = view_t[vidx]
                # combined rotation and translation to map local->camera
                R_comb = R_wc @ R_local
                t_comb = (R_wc @ t_local) + t_wc
                rvec_comb, _ = cv2.Rodrigues(R_comb)
                tvec_comb = t_comb.reshape(3,1)
                X_local = model_pts[pi].reshape(1,3)
                projected, _ = cv2.projectPoints(X_local.astype(np.float64), rvec_comb, tvec_comb, K, dist)
                projected = projected.reshape(2,)
                orig = np.array(view_data[vidx]['points_2d'][pi], dtype=np.float64)
                res.extend((orig - projected).tolist())
            return np.array(res, dtype=np.float64)

        # start from coarse solution
        res_refine = least_squares(residuals_reprojection, x_coarse, loss='soft_l1', f_scale=1.0, verbose=0, max_nfev=500)
        x_ref = res_refine.x
        rvec_ref = x_ref[0:3].reshape(3,)
        t_ref = x_ref[3:6].reshape(3,)
        R_ref, _ = cv2.Rodrigues(rvec_ref)

        # --- Prepare reprojection errors per view/point for output ---
        reprojection_errors = []
        for vidx in range(len(view_data)):
            K = view_K[vidx]
            dist = view_dist[vidx] if view_dist[vidx] is not None else np.zeros(5, dtype=np.float64)
            R_wc = view_R[vidx]
            t_wc = view_t[vidx]
            R_comb = R_wc @ R_ref
            t_comb = (R_wc @ t_ref) + t_wc
            rvec_comb, _ = cv2.Rodrigues(R_comb)
            tvec_comb = t_comb.reshape(3,1)
            view_errors = []
            for pi in range(N):
                orig_uv = view_data[vidx]['points_2d'][pi]
                if orig_uv is None:
                    view_errors.append(None)
                    continue
                X_local = model_pts[pi].reshape(1,3)
                projected, _ = cv2.projectPoints(X_local.astype(np.float64), rvec_comb, tvec_comb, K, dist)
                projected = projected.reshape(2,)
                err = float(np.linalg.norm(np.array(orig_uv, dtype=np.float64) - projected))
                view_errors.append(err)
            reprojection_errors.append(view_errors)

        # local2world matrix
        local2world = np.eye(4, dtype=np.float64)
        local2world[:3, :3] = R_ref
        local2world[:3, 3] = t_ref

        # Transform local points to world coordinates
        points_3d = [(R_ref @ pt + t_ref) for pt in model_pts]

        return {
            'success': True,
            'local2world': local2world,
            'points_3d': points_3d,
            'reprojection_errors': reprojection_errors
        }

    except Exception as e:
        try:
            errmsg = str(e).encode('ascii', 'replace').decode('ascii')
        except Exception:
            errmsg = 'Unknown error'
        return {
            'success': False,
            'error_message': errmsg,
            'num_views': len(view_data) if view_data else 0
        }


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


# ========================================
# PRIVATE HELPER FUNCTIONS
# ========================================


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


def _umeyama(A: np.ndarray, B: np.ndarray, with_scale: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama alignment A -> B. A, B shape (N,3).
    with_scale: estimate similarity (scale) if True. We keep it False (rigid).
    Returns (scale, R, t) where B ~ scale * R @ A + t
    """
    assert A.shape == B.shape and A.ndim == 2 and A.shape[1] == 3
    n = A.shape[0]
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    AA = A - muA
    BB = B - muB
    cov = (BB.T @ AA) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    if with_scale:
        varA = (AA ** 2).sum() / n
        scale = np.trace(np.diag(D) @ S) / varA
    else:
        scale = 1.0
    t = muB - scale * (R @ muA)
    return scale, R, t


def _triangulate_dlt_local(points_2d: List[np.ndarray], proj_mats: List[np.ndarray]) -> np.ndarray:
    """
    DLT triangulation returning 3D point in world coordinates (same as _triangulate_dlt).
    Expects points_2d list of (2,) and proj_mats list of (3,4).
    """
    n_views = len(points_2d)
    A = np.zeros((2 * n_views, 4))
    for i, (pt, P) in enumerate(zip(points_2d, proj_mats)):
        x, y = float(pt[0]), float(pt[1])
        A[2 * i]     = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) > 1e-12:
        return (X[:3] / X[3]).astype(np.float64)
    else:
        return X[:3].astype(np.float64)


def _build_projection_matrices_and_centers(view_data: List[Dict]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build K*[R|t] projection matrices and camera centers from view_data.
    extrinsic is world->camera: X_cam = R * X_world + t
    So camera center in world C = -R^T t
    """
    proj_mats = []
    centers = []
    for view in view_data:
        K = np.array(view['intrinsic'], dtype=np.float64)
        extrinsic = np.array(view['extrinsic'], dtype=np.float64)
        Rm = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        RT = np.hstack([Rm, t.reshape(3, 1)])
        P = K @ RT
        proj_mats.append(P)
        center = -Rm.T @ t
        centers.append(center)
    return proj_mats, centers


def _undistort_points_for_view(points_2d: List[Optional[np.ndarray]], K: np.ndarray, distortion: Optional[np.ndarray]) -> List[Optional[np.ndarray]]:
    """
    Undistort points in pixel coordinates. Returns list same length with None for missing points.
    """
    num = len(points_2d)
    if all(pt is None for pt in points_2d):
        return [None] * num
    valid_idx = [i for i, p in enumerate(points_2d) if p is not None]
    valid_pts = np.array([points_2d[i] for i in valid_idx], dtype=np.float32).reshape(-1, 1, 2)
    if distortion is not None:
        und = cv2.undistortPoints(valid_pts, K, distortion, P=K).reshape(-1, 2)
    else:
        und = valid_pts.reshape(-1, 2)
    out = [None] * num
    for idx, uv in zip(valid_idx, und):
        out[idx] = uv
    return out


def _pixel_to_camera_ray(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    uv: (2,) undistorted pixel coords
    returns direction vector in camera coords (unit)
    """
    x = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    d = np.linalg.inv(K) @ x
    d = d / np.linalg.norm(d)
    return d
