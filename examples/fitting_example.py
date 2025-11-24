"""
Fitting Example Script

This example demonstrates how to use the fitting module to estimate the
rigid transformation (R, t) from local 3D coordinates to world coordinates
using multi-view 2D observations.

The script performs:
1. Intrinsic camera calibration using chessboard images
2. Preparation of view data with 2D observations
3. Multi-view fitting to estimate local->world transformation
4. Quality analysis and visualization

Author: Yizhong Zhang
Date: November 2025
"""

import sys
import glob
import json
from pathlib import Path
import numpy as np
import cv2

# Fix Windows console encoding issues for emoji/Unicode characters
if sys.platform == 'win32':
    try:
        # Set console output encoding to UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass  # Ignore if reconfigure not available

# Add parent directory to path to import the fitting module
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import fitting module (from robot_vision/core)
from core.fitting import fitting_multiview


def load_chessboard_test_data():
    """
    Load chessboard calibration test data for fitting.
    
    This function:
    1. Performs intrinsic calibration using chessboard images
    2. Extracts camera poses and 2D points from calibration
    3. Generates 3D coordinates in the chessboard's local frame
    4. Prepares view data for fitting
    
    Returns:
        tuple: (view_data, target_points_local) or (None, None) if loading failed
            - view_data: List[Dict] - View data ready for fitting
            - target_points_local: List[np.ndarray] - 3D points in local coordinates
        
        Each dict in view_data contains:
        {
            'points_2d': List[np.ndarray or None] - 2D pixel coordinates
            'image_size': tuple - (width, height)
            'intrinsic': np.ndarray - Camera intrinsic matrix (3, 3)
            'distortion': np.ndarray - Distortion coefficients (optional)
            'extrinsic': np.ndarray - World to camera transformation (4, 4)
        }
    """
    # Add camera calibration toolkit to path
    toolkit_path = parent_dir / "ThirdParty" / "camera_calibration_toolkit"
    
    # Need to temporarily manipulate sys.path and modules to avoid conflict with robot_vision's core/
    original_path = sys.path.copy()
    sys.path.insert(0, str(toolkit_path))
    
    # Import calibration toolkit modules
    try:
        # We need to temporarily remove 'core' from sys.modules to allow import from toolkit's core/
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('core.') or k == 'core']
        removed_modules = {}
        for mod in modules_to_remove:
            if mod in sys.modules:
                removed_modules[mod] = sys.modules[mod]
                del sys.modules[mod]
        
        # Now import from toolkit
        from core.intrinsic_calibration import IntrinsicCalibrator
        from core.calibration_patterns import load_pattern_from_json
        
        # Restore our original core modules
        for mod, module in removed_modules.items():
            sys.modules[mod] = module
            
    except ImportError as e:
        # Restore modules even on error
        for mod, module in removed_modules.items():
            sys.modules[mod] = module
        sys.path = original_path
        print(f"Failed to import calibration toolkit: {str(e).encode('ascii', 'replace').decode('ascii')}")
        return None, None
    
    # Load calibration data
    sample_dir = toolkit_path / "sample_data" / "eye_in_hand_test_data"
    image_paths = sorted(glob.glob(str(sample_dir / "*.jpg")))
    
    if not image_paths:
        print(f"No calibration images found in {sample_dir}")
        return None, None
    
    # Load pattern configuration
    config_path = sample_dir / "chessboard_config.json"
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        pattern = load_pattern_from_json(config_data)
    except Exception as e:
        try:
            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        except Exception:
            error_msg = "Unknown error (encoding issue)"
        print(f"Failed to load pattern configuration: {error_msg}")
        return None, None
    
    # Create calibrator and calibrate
    try:
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
    except Exception as e:
        print(f"Calibration failed: {str(e).encode('ascii', 'replace').decode('ascii')}")
        return None, None
    
    camera_matrix = calib_result['camera_matrix']
    distortion_coeffs = calib_result['distortion_coefficients']
    
    # Get detected corners and poses from calibrator
    image_points_all = calibrator.image_points
    rvecs = calibrator.rvecs
    tvecs = calibrator.tvecs
    
    # Filter out None values (failed detections)
    valid_indices = [i for i in range(len(image_points_all)) 
                     if image_points_all[i] is not None]
    
    if len(valid_indices) < 1:
        print(f"Not enough valid views (got {len(valid_indices)}, need at least 1)")
        return None, None
    
    # Generate target 3D points in chessboard local frame
    # The chessboard pattern defines the local coordinate system
    object_points = pattern.generate_object_points()  # Shape: (N, 3)
    
    # Prepare view data
    view_data = []
    
    for idx in valid_indices:
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
        
        # Convert corners_2d to list format for fitting API
        points_2d_list = [corners_2d[i] for i in range(len(corners_2d))]
        
        view_info = {
            'points_2d': points_2d_list,
            'image_size': (img.shape[1], img.shape[0]),
            'intrinsic': camera_matrix,
            'distortion': distortion_coeffs,
            'extrinsic': world2cam
        }
        view_data.append(view_info)
    
    # Convert object points to list of arrays
    target_points_local = [object_points[i] for i in range(len(object_points))]
    
    return view_data, target_points_local


def visualize_fitting_3d(view_data, target_points_local, result):
    """
    Create 3D visualization of fitting results.
    
    Displays:
    - Target points in local frame (blue)
    - Fitted points in world frame (green)
    - Camera positions and orientations
    - Coordinate axes for both frames
    - Quality metrics in the title
    
    Args:
        view_data: List of view dictionaries containing camera parameters
        target_points_local: List of 3D points in local coordinates
        result: Dict from fitting_multiview() with keys:
            - 'success': bool
            - 'local2world': 4x4 transformation matrix
            - 'R': 3x3 rotation matrix
            - 't': 3-element translation vector
            - 'reprojection_errors': List[List] of per-point errors per view
    """
    print("\n[*] Opening 3D visualization...")
    print("-" * 80)
    
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        local2world = result['local2world']
        points_3d = result['points_3d']
        reprojection_errors = result['reprojection_errors']
        
        # Extract R and t from local2world matrix for visualization
        R = local2world[:3, :3]
        t = local2world[:3, 3]
        
        # Convert points_3d list to numpy array
        points_world_fitted = np.array([pt for pt in points_3d], dtype=np.float64)
        
        # Convert target points to numpy array for local frame plot
        points_local = np.array([pt for pt in target_points_local], dtype=np.float64)
        
        # Calculate camera centers from view data
        camera_centers = []
        for view in view_data:
            extrinsic = np.array(view['extrinsic'])
            R_cam = extrinsic[:3, :3]
            t_cam = extrinsic[:3, 3]
            camera_center = -R_cam.T @ t_cam
            camera_centers.append(camera_center)
        camera_centers = np.array(camera_centers)
        
        # Calculate metrics
        num_views = len(view_data)
        num_points = len(target_points_local)
        
        # Calculate mean reprojection error
        valid_errors = []
        for errors in reprojection_errors:
            valid_errors.extend([e for e in errors if e is not None])
        
        mean_reprojection_error = float(np.mean(valid_errors)) if valid_errors else 0.0
        
        fig = plt.figure(figsize=(16, 12))
        
        # Main 3D plot
        ax = fig.add_subplot(221, projection='3d')
        
        # Plot fitted world points
        ax.scatter(points_world_fitted[:, 0], points_world_fitted[:, 1], points_world_fitted[:, 2],
                  c='green', marker='o', s=30, label='Fitted Points (World)', alpha=0.8)
        
        # Plot camera positions
        ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
                  c='red', marker='^', s=150, label='Camera Positions', edgecolors='black', linewidths=2)
        
        # Draw world coordinate axes at fitted location
        origin_world = t
        axis_length = 0.1
        ax.quiver(origin_world[0], origin_world[1], origin_world[2],
                 axis_length, 0, 0, color='r', arrow_length_ratio=0.3, linewidth=2, label='World X')
        ax.quiver(origin_world[0], origin_world[1], origin_world[2],
                 0, axis_length, 0, color='g', arrow_length_ratio=0.3, linewidth=2, label='World Y')
        ax.quiver(origin_world[0], origin_world[1], origin_world[2],
                 0, 0, axis_length, color='b', arrow_length_ratio=0.3, linewidth=2, label='World Z')
        
        # Draw lines from cameras to center of point cloud
        points_center = np.mean(points_world_fitted, axis=0)
        for i, cam_pos in enumerate(camera_centers):
            ax.plot([cam_pos[0], points_center[0]], 
                   [cam_pos[1], points_center[1]], 
                   [cam_pos[2], points_center[2]], 
                   'r--', alpha=0.3, linewidth=1.5)
            # Label cameras
            ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'  Cam{i}', fontsize=10, fontweight='bold')
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        
        title = (f'Multi-View Fitting Result (World Frame)\n{num_points} points from {num_views} views\n'
                f'Mean reprojection error: {mean_reprojection_error:.3f} px')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot local frame
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.scatter(points_local[:, 0], points_local[:, 1], points_local[:, 2],
                   c='blue', marker='o', s=30, label='Target Points (Local)', alpha=0.8)
        
        # Draw local coordinate axes
        ax2.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.3, linewidth=2, label='Local X')
        ax2.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.3, linewidth=2, label='Local Y')
        ax2.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.3, linewidth=2, label='Local Z')
        
        ax2.set_xlabel('X (m)', fontsize=12)
        ax2.set_ylabel('Y (m)', fontsize=12)
        ax2.set_zlabel('Z (m)', fontsize=12)
        ax2.set_title('Target Points (Local Frame)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot transformation parameters
        ax3 = fig.add_subplot(223)
        ax3.axis('off')
        
        # Display rotation matrix
        text_content = "Estimated Transformation (Local → World):\n\n"
        text_content += "Rotation Matrix R:\n"
        for i in range(3):
            text_content += f"  [{R[i,0]:7.4f}  {R[i,1]:7.4f}  {R[i,2]:7.4f}]\n"
        
        text_content += "\nTranslation vector t:\n"
        text_content += f"  [{t[0]:7.4f}  {t[1]:7.4f}  {t[2]:7.4f}]\n"
        
        # Convert rotation to axis-angle and Euler angles for reference
        rvec, _ = cv2.Rodrigues(R)
        angle = np.linalg.norm(rvec)
        if angle > 1e-6:
            axis = rvec.flatten() / angle
            text_content += "\nAxis-Angle Representation:\n"
            text_content += f"  Axis: [{axis[0]:6.3f}, {axis[1]:6.3f}, {axis[2]:6.3f}]\n"
            text_content += f"  Angle: {np.degrees(angle):.2f}°\n"
        
        ax3.text(0.1, 0.5, text_content, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax3.transAxes)
        
        # Plot reprojection error statistics
        ax4 = fig.add_subplot(224)
        
        # Per-view error statistics
        view_mean_errors = []
        for view_idx, errors in enumerate(reprojection_errors):
            valid_view_errors = [e for e in errors if e is not None]
            if valid_view_errors:
                view_mean_errors.append(np.mean(valid_view_errors))
            else:
                view_mean_errors.append(0)
        
        ax4.bar(range(num_views), view_mean_errors, color='steelblue', alpha=0.7)
        ax4.set_xlabel('View Index', fontsize=12)
        ax4.set_ylabel('Mean Reprojection Error (pixels)', fontsize=12)
        ax4.set_title('Per-View Reprojection Error', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticks(range(num_views))
        
        plt.tight_layout()
        plt.show()
        
        print("[+] Visualization closed")
        
    except ImportError as e:
        print(f"[!] Could not create visualization: matplotlib not available ({str(e).encode('ascii', 'replace').decode('ascii')})")
    except Exception as e:
        print(f"[!] Visualization error: {str(e).encode('ascii', 'replace').decode('ascii')}")
        import traceback
        traceback.print_exc()


def run_fitting_example_with_data(view_data, target_points_local, visualize=False):
    """
    Run fitting example using pre-loaded view data and target points.
    
    This function demonstrates:
    1. Multi-view fitting to estimate local->world transformation
    2. Quality analysis (reprojection error)
    3. Optional 3D visualization
    
    Args:
        view_data: List of view dicts from load_chessboard_test_data()
        target_points_local: List of 3D points in local coordinates
        visualize: If True, display 3D visualization
    
    Returns:
        Dict containing test results
    """
    print("\n" + "=" * 80)
    print("Fitting Test")
    print("=" * 80)
    
    try:
        result = fitting_multiview(view_data, target_points_local)
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"[-] Exception in fitting_multiview(): {error_msg}")
        import traceback
        traceback.print_exc()
        return {'error': error_msg}
    
    if not result['success']:
        print(f"[-] Fitting failed: {result.get('error_message', 'Unknown error')}")
        return {'error': result.get('error_message', 'Unknown error')}
    
    local2world = result['local2world']
    points_3d = result['points_3d']
    reprojection_errors = result['reprojection_errors']
    
    # Extract R and t from local2world matrix
    R = local2world[:3, :3]
    t = local2world[:3, 3]
    
    num_points = len(target_points_local)
    num_views = len(view_data)
    
    # Calculate mean reprojection error across all views
    valid_errors = []
    for errors in reprojection_errors:
        valid_errors.extend([e for e in errors if e is not None])
    
    mean_reprojection_error = float(np.mean(valid_errors)) if valid_errors else 0.0
    
    print(f"[+] Fitted transformation for {num_points} 3D points from {num_views} views")
    print(f"   Mean reprojection error: {mean_reprojection_error:.3f} pixels")
    
    # Print per-view errors
    print("\n   Per-view reprojection errors:")
    for view_idx, errors in enumerate(reprojection_errors):
        valid_view_errors = [e for e in errors if e is not None]
        if valid_view_errors:
            print(f"     View {view_idx}: "
                  f"mean={np.mean(valid_view_errors):.3f}px, "
                  f"max={np.max(valid_view_errors):.3f}px, "
                  f"std={np.std(valid_view_errors):.3f}px")
        else:
            print(f"     View {view_idx}: no valid observations")
    
    # Print transformation
    print("\n   Estimated transformation (local → world):")
    print("   Rotation matrix R:")
    for i in range(3):
        print(f"     [{R[i,0]:7.4f}  {R[i,1]:7.4f}  {R[i,2]:7.4f}]")
    print(f"   Translation vector t: [{t[0]:7.4f}  {t[1]:7.4f}  {t[2]:7.4f}]")
    
    # ========================================
    # VISUALIZATION
    # ========================================
    
    if visualize:
        visualize_fitting_3d(view_data, target_points_local, result)
    
    return {
        'num_points': num_points,
        'num_views': num_views,
        'mean_reprojection_error': mean_reprojection_error,
        'R': R,
        't': t,
        'local2world': local2world
    }


def run_fitting_with_missing_detections(view_data, target_points_local, view_visibility=None, visualize=False):
    """
    Test fitting with missing point detections (None values).
    
    This test demonstrates the ability to handle cases where certain points
    are not detected in some views, which is common in real-world scenarios.
    
    Args:
        view_data: List of view dicts from load_chessboard_test_data()
        target_points_local: List of 3D points in local coordinates
        view_visibility: Optional[List[List[int]]] - List of visible point indices for each view.
                        If None, defaults to 4 corner points for all views.
                        Example: [[0, 10, 77, 87], [0, 10, 77, 87], ...] for 4 corners in all views
                        Length must match len(view_data), each inner list contains visible point indices
        visualize: If True, display 3D visualization
    """
    print("\n" + "=" * 80)
    print("Fitting Test with Missing Detections")
    print("=" * 80)
    
    # Create modified view data based on visibility specification
    modified_view_data = []
    num_points = len(target_points_local)
    
    # Default visibility: only 4 corner points
    if view_visibility is None:
        # Determine chessboard dimensions (assume standard 11x8 pattern = 88 points)
        # For a chessboard with width x height corners, the 4 corner indices are:
        # - Top-left: 0
        # - Top-right: width - 1
        # - Bottom-left: width * (height - 1)
        # - Bottom-right: width * height - 1
        
        # Infer width from total points (assuming rectangular pattern)
        # For 88 points: width=11, height=8
        width = 11
        height = num_points // width
        
        corner_indices = [
            0,                          # Top-left
            width - 1,                  # Top-right
            width * (height - 1),       # Bottom-left
            width * height - 1          # Bottom-right
        ]
        
        # Use same corner points for all views
        view_visibility = [corner_indices for _ in range(len(view_data))]
    
    # Validate view_visibility
    if len(view_visibility) != len(view_data):
        raise ValueError(f"view_visibility length ({len(view_visibility)}) must match view_data length ({len(view_data)})")
    
    # Track which points are missing in which views
    missing_matrix = []
    
    for view_idx, view in enumerate(view_data):
        # Convert points_2d to list format (should already be list)
        points_2d_list = list(view['points_2d'])
        
        # Get visible point indices for this view
        visible_indices = set(view_visibility[view_idx])
        
        # Set non-visible points to None
        missing_in_view = [False] * num_points
        for idx in range(num_points):
            if idx not in visible_indices:
                points_2d_list[idx] = None
                missing_in_view[idx] = True
        
        missing_matrix.append(missing_in_view)
        
        # Create modified view
        modified_view = {
            'points_2d': points_2d_list,
            'image_size': view['image_size'],
            'intrinsic': view['intrinsic'],
            'distortion': view['distortion'],
            'extrinsic': view['extrinsic']
        }
        modified_view_data.append(modified_view)
    
    # Count how many views each point appears in
    view_counts = [sum(1 for view_idx in range(len(view_data)) 
                       if not missing_matrix[view_idx][pt_idx])
                   for pt_idx in range(num_points)]
    
    num_points_with_observations = sum(1 for count in view_counts if count >= 1)
    num_points_no_observations = sum(1 for count in view_counts if count == 0)
    
    # Collect unique visible indices across all views
    all_visible_indices = set()
    for visible_list in view_visibility:
        all_visible_indices.update(visible_list)
    
    print("[*] Test scenario:")
    print(f"   - {num_points} points total")
    print(f"   - {len(view_data)} camera views")
    print(f"   - Visible points across all views: {sorted(all_visible_indices)}")
    print(f"   - {num_points_with_observations} points visible in ≥1 views")
    print(f"   - {num_points_no_observations} points visible in 0 views")
    
    # Fit
    print("\n[*] Running fitting...")
    result = fitting_multiview(modified_view_data, target_points_local)
    
    if not result['success']:
        print(f"\n[-] Fitting failed: {result.get('error_message')}")
        return {'error': result.get('error_message')}
    
    reprojection_errors = result['reprojection_errors']
    
    print("[+] Fitting completed")
    
    # Calculate mean error for valid points
    valid_errors = []
    for view_errors in reprojection_errors:
        valid_errors.extend([err for err in view_errors if err is not None])
    
    if valid_errors:
        mean_error = np.mean(valid_errors)
        max_error = np.max(valid_errors)
        print("\n[*] Reprojection error statistics:")
        print(f"   - Mean: {mean_error:.3f} pixels")
        print(f"   - Max: {max_error:.3f} pixels")
    
    # Verify reprojection error structure
    print("\n[*] Checking reprojection errors...")
    
    # Count None values in reprojection errors per view
    none_count_per_view = [sum(1 for err in view_errors if err is None) 
                           for view_errors in reprojection_errors]
    
    for view_idx in range(len(view_data)):
        print(f"   - View {view_idx}: {none_count_per_view[view_idx]} missing points")
    
    if visualize:
        visualize_fitting_3d(modified_view_data, target_points_local, result)
    
    return {
        'num_points_total': num_points,
        'num_points_with_observations': num_points_with_observations,
        'num_points_no_observations': num_points_no_observations,
        'mean_reprojection_error': mean_error if valid_errors else 0.0
    }


def test_error_handling():
    """
    Test error handling with various invalid inputs.
    
    Tests:
    1. Missing required fields
    2. Mismatched point counts
    3. Empty point arrays
    4. Invalid matrix shapes
    5. Insufficient views
    """
    print("\n" + "="*80)
    print("ERROR HANDLING TESTS")
    print("="*80)
    
    # Create some dummy 3D target points
    target_points_local = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0])
    ]
    
    test_cases = []
    
    # Test 1: Missing required field
    test_cases.append({
        'name': 'Missing intrinsic matrix',
        'view_data': [
            {
                'points_2d': [np.array([100, 100]), np.array([200, 200]), np.array([150, 150])],
                'image_size': (640, 480),
                # Missing 'intrinsic'
                'extrinsic': np.eye(4, dtype=np.float64)
            }
        ],
        'target_points': target_points_local
    })
    
    # Test 2: Mismatched point counts
    test_cases.append({
        'name': 'Mismatched point counts',
        'view_data': [
            {
                'points_2d': [np.array([100, 100]), np.array([200, 200])],  # Only 2 points
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.eye(4, dtype=np.float64)
            }
        ],
        'target_points': target_points_local  # 3 points
    })
    
    # Test 3: Empty point arrays
    test_cases.append({
        'name': 'Empty point arrays',
        'view_data': [
            {
                'points_2d': [],
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.eye(4, dtype=np.float64)
            }
        ],
        'target_points': []
    })
    
    # Test 4: None input
    test_cases.append({
        'name': 'None input',
        'view_data': None,
        'target_points': target_points_local
    })
    
    # Test 5: No observations (all None)
    test_cases.append({
        'name': 'No observations (all points None)',
        'view_data': [
            {
                'points_2d': [None, None, None],
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.eye(4, dtype=np.float64)
            }
        ],
        'target_points': target_points_local
    })
    
    # Run all tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 70)
        
        try:
            result = fitting_multiview(test_case['view_data'], test_case['target_points'])
            
            if result['success']:
                print("  Result: SUCCESS")
            else:
                print("  Result: FAILED (as expected)")
                print(f"  Error: {result.get('error_message', 'No error message')}")
        except Exception as e:
            print("  Result: EXCEPTION")
            print(f"  Exception: {type(e).__name__}: {str(e).encode('ascii', 'replace').decode('ascii')}")
    
    print("\n" + "="*80)
    print("ERROR HANDLING TESTS COMPLETED")
    print("="*80)


def main():
    """
    Main function to run fitting examples.
    
    Parses command-line arguments and runs tests with the specified configuration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-View Fitting Example using Chessboard Calibration Data'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Enable 3D visualization of fitting results'
    )
    
    args = parser.parse_args()
    
    print("Multi-View Fitting Example")
    print("=" * 80)
    print("\nThis example demonstrates how to estimate rigid transformation from local")
    print("to world coordinates using multi-view 2D observations.")
    print("=" * 80)
    
    # Load test data
    print("\n[*] Loading test data...")
    print("-" * 80)
    
    view_data, target_points_local = load_chessboard_test_data()
    
    if view_data is None or target_points_local is None:
        print("\n[-] Failed to load test data")
        print("\n[!] The camera calibration toolkit submodule may not be initialized.")
        print("   Please run the following commands to update the submodule:")
        print("   git submodule update --init --recursive")
        sys.exit(1)
    
    print("[+] Test data loaded successfully")
    print(f"   - {len(view_data)} valid views")
    print(f"   - {len(target_points_local)} 3D target points")
    print(f"   - Image size: {view_data[0]['image_size']}")
    
    # Run fitting examples
    try:
        print("\n" + "=" * 80)
        print("Running Fitting Examples")
        print("=" * 80)
        
        # Test 1: Standard fitting with all observations
        example_result = run_fitting_example_with_data(
            view_data=view_data,
            target_points_local=target_points_local,
            visualize=args.visualize
        )
        
        if 'error' in example_result:
            print(f"\n[!] Test 1 completed with issues: {example_result['error']}")
        else:
            print("\n[+] Test 1 completed successfully!")
            print("\nTest 1 Results:")
            print(f"  - Fitted {example_result['num_points']} 3D points")
            print(f"  - Used {example_result['num_views']} camera views")
            print(f"  - Reprojection error: {example_result['mean_reprojection_error']:.3f} pixels")
        
        # Test 2: Fitting with missing detections (default: 4 corners in all views)
        missing_detection_result = run_fitting_with_missing_detections(
            view_data=view_data,
            target_points_local=target_points_local,
            visualize=args.visualize
        )
        
        if 'error' in missing_detection_result:
            print(f"\n[!] Test 2 completed with issues: {missing_detection_result['error']}")
        else:
            print("\n[+] Test 2 completed successfully!")
            print("\nTest 2 Results:")
            print(f"  - Total points: {missing_detection_result['num_points_total']}")
            print(f"  - Points with observations: {missing_detection_result['num_points_with_observations']}")
            print(f"  - Points with no observations: {missing_detection_result['num_points_no_observations']}")
            print(f"  - Mean reprojection error: {missing_detection_result['mean_reprojection_error']:.3f} pixels")
        
        # Test 3: Fitting with diverse visibility patterns
        # View 0, 1: top corners visible (indices 0, 10)
        # View 2, 3: bottom left corner visible (index 77)
        # View 4, 5: none visible (empty list)
        num_points = len(target_points_local)
        width = 11
        height = num_points // width
        
        diverse_visibility = [
            [0, width - 1],                     # View 0: top corners (0, 10)
            [0, width - 1],                     # View 1: top corners (0, 10)
            [width * (height - 1)],             # View 2: bottom left corner (77)
            [width * (height - 1)],             # View 3: bottom left corner (77)
            [],                                  # View 4: none visible
            []                                   # View 5: none visible
        ]
        
        diverse_result = run_fitting_with_missing_detections(
            view_data=view_data,
            target_points_local=target_points_local,
            view_visibility=diverse_visibility,
            visualize=args.visualize
        )
        
        if 'error' in diverse_result:
            print(f"\n[!] Test 3 completed with issues: {diverse_result['error']}")
        else:
            print("\n[+] Test 3 completed successfully!")
            print("\nTest 3 Results:")
            print(f"  - Total points: {diverse_result['num_points_total']}")
            print(f"  - Points with observations: {diverse_result['num_points_with_observations']}")
            print(f"  - Points with no observations: {diverse_result['num_points_no_observations']}")
            print(f"  - Mean reprojection error: {diverse_result['mean_reprojection_error']:.3f} pixels")
        
        # Test 4: Fitting with 4 corners distributed across views
        # View 0, 1: top corners visible (indices 0, 10)
        # View 2: bottom left corner visible (index 77)
        # View 3: bottom right corner visible (index 87)
        # View 4, 5: use default (will be truncated or repeated as needed)
        distributed_visibility = [
            [0, width - 1],                     # View 0: top corners (0, 10)
            [0, width - 1],                     # View 1: top corners (0, 10)
            [width * (height - 1)],             # View 2: bottom left corner (77)
            [width * height - 1],               # View 3: bottom right corner (87)
        ]
        
        # Extend to match number of views
        while len(distributed_visibility) < len(view_data):
            distributed_visibility.append([])
        
        distributed_result = run_fitting_with_missing_detections(
            view_data=view_data,
            target_points_local=target_points_local,
            view_visibility=distributed_visibility,
            visualize=args.visualize
        )
        
        if 'error' in distributed_result:
            print(f"\n[!] Test 4 completed with issues: {distributed_result['error']}")
        else:
            print("\n[+] Test 4 completed successfully!")
            print("\nTest 4 Results:")
            print(f"  - Total points: {distributed_result['num_points_total']}")
            print(f"  - Points with observations: {distributed_result['num_points_with_observations']}")
            print(f"  - Points with no observations: {distributed_result['num_points_no_observations']}")
            print(f"  - Mean reprojection error: {distributed_result['mean_reprojection_error']:.3f} pixels")
        
        # Test 5: Fitting with 3 corners, one corner per view
        # View 0, 1: top left corner (index 0)
        # View 2: top right corner (index 10)
        # View 3: bottom left corner (index 77)
        # View 4, 5: none visible
        three_corners_visibility = [
            [0],                                 # View 0: top left corner
            [0],                                 # View 1: top left corner
            [width - 1],                         # View 2: top right corner (10)
            [width * (height - 1)],              # View 3: bottom left corner (77)
        ]
        
        # Extend to match number of views
        while len(three_corners_visibility) < len(view_data):
            three_corners_visibility.append([])
        
        three_corners_result = run_fitting_with_missing_detections(
            view_data=view_data,
            target_points_local=target_points_local,
            view_visibility=three_corners_visibility,
            visualize=args.visualize
        )
        
        if 'error' in three_corners_result:
            print(f"\n[!] Test 5 completed with issues: {three_corners_result['error']}")
        else:
            print("\n[+] Test 5 completed successfully!")
            print("\nTest 5 Results:")
            print(f"  - Total points: {three_corners_result['num_points_total']}")
            print(f"  - Points with observations: {three_corners_result['num_points_with_observations']}")
            print(f"  - Points with no observations: {three_corners_result['num_points_no_observations']}")
            print(f"  - Mean reprojection error: {three_corners_result['mean_reprojection_error']:.3f} pixels")
        
        # Test 6: Fitting with 4 corners, each in a different view
        # View 0: top left corner (index 0)
        # View 1: top right corner (index 10)
        # View 2: bottom left corner (index 77)
        # View 3: bottom right corner (index 87)
        # View 4, 5: none visible
        single_corner_per_view = [
            [0],                                 # View 0: top left corner
            [width - 1],                         # View 1: top right corner (10)
            [width * (height - 1)],              # View 2: bottom left corner (77)
            [width * height - 1],                # View 3: bottom right corner (87)
        ]
        
        # Extend to match number of views
        while len(single_corner_per_view) < len(view_data):
            single_corner_per_view.append([])
        
        single_corner_result = run_fitting_with_missing_detections(
            view_data=view_data,
            target_points_local=target_points_local,
            view_visibility=single_corner_per_view,
            visualize=args.visualize
        )
        
        if 'error' in single_corner_result:
            print(f"\n[!] Test 6 completed with issues: {single_corner_result['error']}")
        else:
            print("\n[+] Test 6 completed successfully!")
            print("\nTest 6 Results:")
            print(f"  - Total points: {single_corner_result['num_points_total']}")
            print(f"  - Points with observations: {single_corner_result['num_points_with_observations']}")
            print(f"  - Points with no observations: {single_corner_result['num_points_no_observations']}")
            print(f"  - Mean reprojection error: {single_corner_result['mean_reprojection_error']:.3f} pixels")
        
        # Test 7: Fitting with 3 corners starting from view 1
        # View 0: none visible
        # View 1: top left corner (index 0)
        # View 2: top right corner (index 10)
        # View 3: bottom left corner (index 77)
        # View 4, 5: none visible
        three_corners_shifted = [
            [],                                  # View 0: none visible
            [0],                                 # View 1: top left corner
            [width - 1],                         # View 2: top right corner (10)
            [width * (height - 1)],              # View 3: bottom left corner (77)
        ]
        
        # Extend to match number of views
        while len(three_corners_shifted) < len(view_data):
            three_corners_shifted.append([])
        
        three_corners_shifted_result = run_fitting_with_missing_detections(
            view_data=view_data,
            target_points_local=target_points_local,
            view_visibility=three_corners_shifted,
            visualize=args.visualize
        )
        
        if 'error' in three_corners_shifted_result:
            print(f"\n[!] Test 7 completed with issues: {three_corners_shifted_result['error']}")
        else:
            print("\n[+] Test 7 completed successfully!")
            print("\nTest 7 Results:")
            print(f"  - Total points: {three_corners_shifted_result['num_points_total']}")
            print(f"  - Points with observations: {three_corners_shifted_result['num_points_with_observations']}")
            print(f"  - Points with no observations: {three_corners_shifted_result['num_points_no_observations']}")
            print(f"  - Mean reprojection error: {three_corners_shifted_result['mean_reprojection_error']:.3f} pixels")
        
        # Test 8: Error handling tests
        test_error_handling()
        
        sys.exit(0)
    except Exception as e:
        print(f"\n[-] Example failed with exception: {str(e).encode('ascii', 'replace').decode('ascii')}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
