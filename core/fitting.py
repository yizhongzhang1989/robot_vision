import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from scipy.optimize import least_squares

# -------------------------
# Helper utilities
# -------------------------
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

# -------------------------
# Main function
# -------------------------
def fitting_multiview(view_data: List[Dict], target_point3d_local: List[np.ndarray]) -> Dict:
    """
    Estimate local->world rigid transform (R, t) such that world = R * local + t.
    Uses world->camera extrinsic convention provided in view_data.

    Returns dict with keys:
      'success': bool
      'local2world': 4x4 np.ndarray - transformation matrix from local to world frame
      'points_3d': List[np.ndarray] - 3D points in world coordinates (N points of shape (3,))
      'reprojection_errors': list (per view) of lists (per point) of float or None
      'error_message': optional (only present when success=False)
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

        # --- undistort points for each view and prepare projection matrices
        undistorted_views = []
        proj_mats, cam_centers = _build_projection_matrices_and_centers(view_data)
        # also prepare per-view extrinsic R,t and K, distortion
        view_K = []
        view_R = []
        view_t = []
        view_dist = []
        for v in view_data:
            K = np.array(v['intrinsic'], dtype=np.float64)
            extrinsic = np.array(v['extrinsic'], dtype=np.float64)
            Rm = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            dist = np.array(v['distortion'], dtype=np.float64) if 'distortion' in v else None
            und = _undistort_points_for_view(v['points_2d'], K, dist)
            undistorted_views.append(und)
            view_K.append(K)
            view_R.append(Rm)
            view_t.append(t)
            view_dist.append(dist)

        # --- prepare "rays" in world coordinates for each observation
        # rays_per_obs: list (view) of lists (per point) of either None or (O_world, v_world)
        rays_per_view = []
        for vidx, und in enumerate(undistorted_views):
            K = view_K[vidx]
            R_wc = view_R[vidx]  # world->camera
            t_wc = view_t[vidx]
            O_world = -R_wc.T @ t_wc
            view_rays = []
            for p in und:
                if p is None:
                    view_rays.append(None)
                else:
                    d_cam = _pixel_to_camera_ray(np.asarray(p, dtype=np.float64), K)
                    v_world = R_wc.T @ d_cam
                    v_world = v_world / np.linalg.norm(v_world)
                    view_rays.append((O_world.copy(), v_world.copy()))
            rays_per_view.append(view_rays)

        # --- TRY to triangulate model points that have >=2 observations (to get world points)
        triangulated_world = [None] * N
        for pi in range(N):
            pts2d = []
            Pmats = []
            for vidx in range(len(view_data)):
                uv = undistorted_views[vidx][pi]
                if uv is not None:
                    pts2d.append(uv)
                    Pmats.append(proj_mats[vidx])
            if len(pts2d) >= 2:
                try:
                    Xw = _triangulate_dlt_local(pts2d, Pmats)
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
        t_coarse = x_coarse[3:6]
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
