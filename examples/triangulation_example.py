"""
Triangulation Example Script

This example demonstrates how to use the triangulation module to reconstruct
3D points from multiple camera views using chessboard calibration data.

The script performs:
1. Intrinsic camera calibration using chessboard images
2. Preparation of view data from multiple camera poses
3. Multi-view triangulation of 3D points
4. Quality analysis and visualization

Author: Yizhong Zhang
Date: November 2025
"""

import sys
import os
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

# Add parent directory to path to import the triangulation module
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import triangulation module (from robot_vision/core)
from core.triangulation import triangulate_multiview


def load_chessboard_test_data():
    """
    Load chessboard calibration test data for triangulation.
    
    This function:
    1. Performs intrinsic calibration using chessboard images
    2. Extracts camera poses and 2D points from calibration
    3. Prepares view data for triangulation
    
    Returns:
        tuple: (view_data, pattern) or (None, None) if loading failed
            - view_data: List[Dict] - View data ready for triangulation
            - pattern: Pattern object - Chessboard pattern configuration
        
        Each dict in view_data contains:
        {
            'points_2d': np.ndarray - 2D pixel coordinates (N, 2)
            'image_size': tuple - (width, height)
            'intrinsic': np.ndarray - Camera intrinsic matrix (3, 3)
            'distortion': np.ndarray - Distortion coefficients
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
        return None
    
    # Load calibration data
    sample_dir = toolkit_path / "sample_data" / "eye_in_hand_test_data"
    image_paths = sorted(glob.glob(str(sample_dir / "*.jpg")))
    
    if not image_paths:
        print(f"No calibration images found in {sample_dir}")
        return None
    
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
        return None
    
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
        return None
    
    camera_matrix = calib_result['camera_matrix']
    distortion_coeffs = calib_result['distortion_coefficients']
    
    # Get detected corners and poses from calibrator
    image_points_all = calibrator.image_points
    rvecs = calibrator.rvecs
    tvecs = calibrator.tvecs
    
    # Filter out None values (failed detections)
    valid_indices = [i for i in range(len(image_points_all)) 
                     if image_points_all[i] is not None]
    
    if len(valid_indices) < 2:
        print(f"Not enough valid views (got {len(valid_indices)}, need at least 2)")
        return None
    
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
        
        view_info = {
            'points_2d': corners_2d,
            'image_size': (img.shape[1], img.shape[0]),
            'intrinsic': camera_matrix,
            'distortion': distortion_coeffs,
            'extrinsic': world2cam
        }
        view_data.append(view_info)
    
    return view_data


def visualize_triangulation_3d(view_data, result):
    """
    Create 3D visualization of triangulation results.
    
    Displays:
    - Triangulated 3D points
    - Camera positions and orientations
    - Lines connecting cameras to point cloud center
    - Quality metrics in the title
    
    Args:
        view_data: List of view dictionaries containing camera parameters
        result: Dict from triangulate_multiview() with keys:
            - 'success': bool
            - 'points_3d': np.ndarray of triangulated 3D points
            - 'reprojection_errors': List[np.ndarray] of per-point errors
    """
    print("\n[*] Opening 3D visualization...")
    print("-" * 80)
    
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        points_3d = result['points_3d']
        reprojection_errors = result['reprojection_errors']
        
        # Calculate camera centers from view data
        camera_centers = []
        for view in view_data:
            extrinsic = np.array(view['extrinsic'])
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            camera_center = -R.T @ t
            camera_centers.append(camera_center)
        camera_centers = np.array(camera_centers)
        
        # Calculate metrics
        num_points = len(points_3d)
        num_views = len(view_data)
        mean_reprojection_error = float(np.mean([np.mean(errors) for errors in reprojection_errors]))
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot triangulated 3D points
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='blue', marker='o', s=30, label='Triangulated Points', alpha=0.8)
        
        # Plot camera positions
        ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
                  c='red', marker='^', s=150, label='Camera Positions', edgecolors='black', linewidths=2)
        
        # Draw lines from cameras to center of point cloud
        points_center = np.mean(points_3d, axis=0)
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
        ax.set_title(f'3D Triangulation Result\n{num_points} points from {num_views} views\n'
                    f'Mean reprojection error: {mean_reprojection_error:.3f} px',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        
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
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("[+] Visualization closed")
        
    except ImportError as e:
        print(f"[!] Could not create visualization: matplotlib not available ({str(e).encode('ascii', 'replace').decode('ascii')})")
    except Exception as e:
        print(f"[!] Visualization error: {str(e).encode('ascii', 'replace').decode('ascii')}")
        import traceback
        traceback.print_exc()


def run_triangulation_example_with_data(view_data, visualize=False):
    """
    Run triangulation example using pre-loaded view data.
    
    This function demonstrates:
    1. Multi-view triangulation of 3D points
    2. Quality analysis (reprojection error, planarity, spacing)
    3. Optional 3D visualization
    
    Args:
        view_data: List of view dicts from load_chessboard_test_data()
        visualize: If True, display 3D visualization of triangulated points and cameras
    
    Returns:
        Dict containing test results
    """
    print("\n" + "=" * 80)
    print("Triangulation Test")
    print("=" * 80)
    
    try:
        result = triangulate_multiview(view_data)
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"[-] Exception in triangulate_multiview(): {error_msg}")
        import traceback
        traceback.print_exc()
        return {'error': error_msg}
    
    if not result['success']:
        print(f"[-] Triangulation failed: {result.get('error_message', 'Unknown error')}")
        return {'error': result.get('error_message', 'Unknown error')}
    
    points_3d = result['points_3d']
    reprojection_errors = result['reprojection_errors']  # List of per-point errors for each view
    
    num_points = len(points_3d)
    num_views = len(view_data)
    
    # Calculate mean reprojection error across all views
    all_errors = np.concatenate(reprojection_errors)
    mean_reprojection_error = float(np.mean(all_errors))
    
    print(f"[+] Triangulated {num_points} 3D points from {num_views} views")
    print(f"   Mean reprojection error: {mean_reprojection_error:.3f} pixels")
    
    # Print per-view errors
    print("\n   Per-view reprojection errors:")
    for view_idx, errors in enumerate(reprojection_errors):
        print(f"     View {view_idx}: "
              f"mean={np.mean(errors):.3f}px, "
              f"max={np.max(errors):.3f}px, "
              f"std={np.std(errors):.3f}px")
    
    # ========================================
    # VISUALIZATION
    # ========================================
    
    if visualize:
        visualize_triangulation_3d(view_data, result)
    
    return {
        'num_points': num_points,
        'num_views': num_views,
        'mean_reprojection_error': mean_reprojection_error,
        'points_3d': points_3d
    }


def run_triangulation_with_undistorted_points(view_data, visualize=False):
    """
    Run triangulation with first view using pre-undistorted 2D points (no distortion key).
    
    This function demonstrates triangulation workflow where the first view has pre-undistorted
    points without distortion coefficients, while other views use standard distortion model.
    
    Workflow:
    1. Undistort 2D points for the first view only
    2. Create modified view data where first view has no 'distortion' key
    3. Other views remain unchanged with original distortion
    4. Triangulate using the mixed view data
    
    Args:
        view_data: List of view dicts from load_chessboard_test_data()
        visualize: If True, display 3D visualization of triangulated points and cameras
    
    Returns:
        Dict containing test results
    """
    print("\n" + "=" * 80)
    print("Triangulation Test (First View Pre-undistorted)")
    print("=" * 80)
    
    # Create modified view data with first two views undistorted
    mixed_view_data = []
    
    for idx, view in enumerate(view_data):
        if idx == 0:
            # Undistort first view
            points_2d = np.array(view['points_2d'], dtype=np.float32)
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            distortion = np.array(view['distortion'], dtype=np.float64)
            
            # Undistort points
            points_reshaped = points_2d.reshape(-1, 1, 2)
            undistorted_points = cv2.undistortPoints(
                points_reshaped,
                intrinsic,
                distortion,
                P=intrinsic
            )
            undistorted_2d = undistorted_points.reshape(-1, 2)
            
            # Create view without distortion key
            undistorted_view = {
                'points_2d': undistorted_2d,
                'image_size': view['image_size'],
                'intrinsic': intrinsic,
                # No 'distortion' key - indicates pre-undistorted points
                'extrinsic': view['extrinsic']
            }
            
            mixed_view_data.append(undistorted_view)
        elif idx == 1:
            # Undistort second view and resize to half
            points_2d = np.array(view['points_2d'], dtype=np.float32)
            intrinsic = np.array(view['intrinsic'], dtype=np.float64)
            distortion = np.array(view['distortion'], dtype=np.float64)
            
            # Undistort points
            points_reshaped = points_2d.reshape(-1, 1, 2)
            undistorted_points = cv2.undistortPoints(
                points_reshaped,
                intrinsic,
                distortion,
                P=intrinsic
            )
            undistorted_2d = undistorted_points.reshape(-1, 2)
            
            # Scale points to half resolution
            scale_factor = 0.5
            scaled_points = undistorted_2d * scale_factor
            
            # Scale intrinsic matrix
            scaled_intrinsic = intrinsic.copy()
            scaled_intrinsic[0, 0] *= scale_factor  # fx
            scaled_intrinsic[1, 1] *= scale_factor  # fy
            scaled_intrinsic[0, 2] *= scale_factor  # cx
            scaled_intrinsic[1, 2] *= scale_factor  # cy
            
            # Scale image size
            original_width, original_height = view['image_size']
            scaled_image_size = (int(original_width * scale_factor), int(original_height * scale_factor))
            
            # Create view without distortion key and with scaled parameters
            scaled_view = {
                'points_2d': scaled_points,
                'image_size': scaled_image_size,
                'intrinsic': scaled_intrinsic,
                # No 'distortion' key - indicates pre-undistorted points
                'extrinsic': view['extrinsic']
            }
            
            mixed_view_data.append(scaled_view)
        else:
            # Keep other views unchanged
            mixed_view_data.append(view)
    
    print(f"[+] Modified view data:")
    print(f"   - View 0: pre-undistorted (no distortion key)")
    print(f"   - View 1: pre-undistorted + resized to half (no distortion key)")
    print(f"   - Views 2-{len(view_data)-1}: standard with distortion model")
    
    # Triangulate with mixed view data
    try:
        result = triangulate_multiview(mixed_view_data)
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"[-] Exception in triangulate_multiview(): {error_msg}")
        import traceback
        traceback.print_exc()
        return {'error': error_msg}
    
    if not result['success']:
        print(f"[-] Triangulation failed: {result.get('error_message', 'Unknown error')}")
        return {'error': result.get('error_message', 'Unknown error')}
    
    points_3d = result['points_3d']
    reprojection_errors = result['reprojection_errors']
    
    num_points = len(points_3d)
    num_views = len(mixed_view_data)
    
    # Calculate mean reprojection error across all views
    all_errors = np.concatenate(reprojection_errors)
    mean_reprojection_error = float(np.mean(all_errors))
    
    print(f"[+] Triangulated {num_points} 3D points from {num_views} views")
    print(f"   Mean reprojection error: {mean_reprojection_error:.3f} pixels")
    
    # Print per-view errors
    print("\n   Per-view reprojection errors:")
    for view_idx, errors in enumerate(reprojection_errors):
        if view_idx == 0:
            view_type = "(pre-undistorted)"
        elif view_idx == 1:
            view_type = "(pre-undistorted + half size)"
        else:
            view_type = "(with distortion)"
        print(f"     View {view_idx} {view_type}: "
              f"mean={np.mean(errors):.3f}px, "
              f"max={np.max(errors):.3f}px, "
              f"std={np.std(errors):.3f}px")
    
    if visualize:
        visualize_triangulation_3d(mixed_view_data, result)
    
    return {
        'num_points': num_points,
        'num_views': num_views,
        'mean_reprojection_error': mean_reprojection_error,
        'points_3d': points_3d
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
    
    test_cases = []
    
    # Test 1: Missing required field
    test_cases.append({
        'name': 'Missing intrinsic matrix',
        'data': [
            {
                'points_2d': np.array([[100, 100], [200, 200]], dtype=np.float32),
                'image_size': (640, 480),
                # Missing 'intrinsic'
                'extrinsic': np.eye(4, dtype=np.float64)
            },
            {
                'points_2d': np.array([[150, 100], [250, 200]], dtype=np.float32),
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
            }
        ]
    })
    
    # Test 2: Mismatched point counts
    test_cases.append({
        'name': 'Mismatched point counts between views',
        'data': [
            {
                'points_2d': np.array([[100, 100], [200, 200]], dtype=np.float32),
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.eye(4, dtype=np.float64)
            },
            {
                'points_2d': np.array([[150, 100], [250, 200], [300, 300]], dtype=np.float32),  # 3 points
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
            }
        ]
    })
    
    # Test 3: Empty point arrays
    test_cases.append({
        'name': 'Empty point arrays',
        'data': [
            {
                'points_2d': np.array([], dtype=np.float32).reshape(0, 2),
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.eye(4, dtype=np.float64)
            },
            {
                'points_2d': np.array([], dtype=np.float32).reshape(0, 2),
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
            }
        ]
    })
    
    # Test 4: Too few views
    test_cases.append({
        'name': 'Only one view (insufficient)',
        'data': [
            {
                'points_2d': np.array([[100, 100], [200, 200]], dtype=np.float32),
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.eye(4, dtype=np.float64)
            }
        ]
    })
    
    # Test 5: None input
    test_cases.append({
        'name': 'None input',
        'data': None
    })
    
    # Test 6: Invalid intrinsic shape
    test_cases.append({
        'name': 'Invalid intrinsic matrix shape (2x2 instead of 3x3)',
        'data': [
            {
                'points_2d': np.array([[100, 100], [200, 200]], dtype=np.float32),
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0], [0, 800]], dtype=np.float64),  # Wrong shape
                'extrinsic': np.eye(4, dtype=np.float64)
            },
            {
                'points_2d': np.array([[150, 100], [250, 200]], dtype=np.float32),
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
            }
        ]
    })
    
    # Test 7: Identical camera poses (degenerate)
    test_cases.append({
        'name': 'Identical camera poses (degenerate configuration)',
        'data': [
            {
                'points_2d': np.array([[320, 240], [350, 250]], dtype=np.float32),
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.eye(4, dtype=np.float64)
            },
            {
                'points_2d': np.array([[320, 240], [350, 250]], dtype=np.float32),
                'image_size': (640, 480),
                'intrinsic': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64),
                'extrinsic': np.eye(4, dtype=np.float64)  # Same pose
            }
        ]
    })
    
    # Run all tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 70)
        
        try:
            result = triangulate_multiview(test_case['data'])
            
            if result['success']:
                print(f"  Result: SUCCESS")
                if 'points_3d' in result:
                    print(f"  Triangulated {len(result['points_3d'])} points")
            else:
                print(f"  Result: FAILED (as expected)")
                print(f"  Error: {result.get('error_message', 'No error message')}")
        except Exception as e:
            print(f"  Result: EXCEPTION")
            print(f"  Exception: {type(e).__name__}: {str(e).encode('ascii', 'replace').decode('ascii')}")
    
    print("\n" + "="*80)
    print("ERROR HANDLING TESTS COMPLETED")
    print("="*80)


def main():
    """
    Main function to run triangulation examples.
    
    Parses command-line arguments and runs tests with the specified configuration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-View Triangulation Example using Chessboard Calibration Data'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Enable 3D visualization of triangulation results'
    )
    
    args = parser.parse_args()
    
    print("Multi-View Triangulation Example")
    print("=" * 80)
    print("\nThis example demonstrates how to triangulate 3D points from multiple views")
    print("using chessboard calibration data.")
    print("=" * 80)
    
    # Load test data once
    print("\n[*] Loading test data...")
    print("-" * 80)
    
    view_data = load_chessboard_test_data()
    
    if view_data is None:
        print("\n[-] Failed to load test data")
        print("\n[!] The camera calibration toolkit submodule may not be initialized.")
        print("   Please run the following commands to update the submodule:")
        print("   git submodule update --init --recursive")
        sys.exit(1)
    
    print("[+] Test data loaded successfully")
    print(f"   - {len(view_data)} valid views")
    print(f"   - {len(view_data[0]['points_2d'])} points per view")
    print(f"   - Image size: {view_data[0]['image_size']}")
    
    # Run triangulation examples
    try:
        print("\n" + "=" * 80)
        print("Running Triangulation Examples")
        print("=" * 80)
        
        # Test 1: Standard triangulation with distortion model
        example_result = run_triangulation_example_with_data(
            view_data=view_data,
            visualize=args.visualize
        )
        
        if 'error' in example_result:
            print(f"\n[!] Test 1 completed with issues: {example_result['error']}")
        else:
            print("\n[+] Test 1 completed successfully!")
            print("\nTest 1 Results:")
            print(f"  - Triangulated {example_result['num_points']} 3D points")
            print(f"  - Used {example_result['num_views']} camera views")
            print(f"  - Reprojection error: {example_result['mean_reprojection_error']:.3f} pixels")
        
        # Test 2: Triangulation with pre-undistorted points (no distortion model)
        undistorted_result = run_triangulation_with_undistorted_points(
            view_data=view_data,
            visualize=args.visualize
        )
        
        if 'error' in undistorted_result:
            print(f"\n[!] Test 2 completed with issues: {undistorted_result['error']}")
        else:
            print("\n[+] Test 2 completed successfully!")
            print("\nTest 2 Results:")
            print(f"  - Triangulated {undistorted_result['num_points']} 3D points")
            print(f"  - Used {undistorted_result['num_views']} camera views")
            print(f"  - Reprojection error: {undistorted_result['mean_reprojection_error']:.3f} pixels")
        
        # Summary
        if 'error' not in example_result and 'error' not in undistorted_result:
            print("\n" + "=" * 80)
            print("COMPARISON SUMMARY")
            print("=" * 80)
            print(f"\nTest 1 (with distortion model):     {example_result['mean_reprojection_error']:.3f} px")
            print(f"Test 2 (pre-undistorted, no model):  {undistorted_result['mean_reprojection_error']:.3f} px")
            print(f"Difference:                          {abs(example_result['mean_reprojection_error'] - undistorted_result['mean_reprojection_error']):.3f} px")
        
        # Test 3: Error handling tests
        test_error_handling()
        
        sys.exit(0)
    except Exception as e:
        print(f"\n[-] Example failed with exception: {str(e).encode('ascii', 'replace').decode('ascii')}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
