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
import glob
import json
from pathlib import Path
import numpy as np
import cv2

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
        print(f"Failed to import calibration toolkit: {e}")
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
        print(f"Failed to load pattern configuration: {e}")
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
        print(f"Calibration failed: {e}")
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


def visualize_triangulation_3d(view_data, triangulation_result):
    """
    Create 3D visualization of triangulation results.
    
    Displays:
    - Triangulated 3D points
    - Camera positions and orientations
    - Lines connecting cameras to point cloud center
    - Quality metrics in the title
    
    Args:
        view_data: List of view dictionaries containing camera parameters
        triangulation_result: Dict containing triangulation results with keys:
            - 'points_3d': np.ndarray of triangulated 3D points
            - 'camera_centers': np.ndarray of camera center positions
            - 'num_points': Number of triangulated points
            - 'num_views': Number of views used
            - 'mean_reprojection_error': Mean reprojection error in pixels
    """
    print("\n[*] Opening 3D visualization...")
    print("-" * 80)
    
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        points_3d = triangulation_result['points_3d']
        camera_centers = triangulation_result['camera_centers']
        
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
        ax.set_title(f'3D Triangulation Result\n{triangulation_result["num_points"]} points from {triangulation_result["num_views"]} views\n'
                    f'Mean reprojection error: {triangulation_result["mean_reprojection_error"]:.3f} px',
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
        print(f"[!] Could not create visualization: matplotlib not available ({e})")
    except Exception as e:
        print(f"[!] Visualization error: {e}")
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
    
    # ========================================
    # STEP 1: DATA SUMMARY
    # ========================================
    
    print("\n[1] Step 1: View data summary...")
    print("-" * 80)
    
    print(f"Number of views: {len(view_data)}")
    print(f"Points per view: {len(view_data[0]['points_2d'])}")
    print(f"Image size: {view_data[0]['image_size']}")
    
    # ========================================
    # STEP 2: TRIANGULATE 3D POINTS
    # ========================================
    
    print("\n[2] Step 2: Triangulating 3D points...")
    print("-" * 80)
    
    try:
        result = triangulate_multiview(view_data)
    except Exception as e:
        print(f"[-] Exception in triangulate_multiview(): {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
    
    if not result['success']:
        print(f"[-] Triangulation failed: {result.get('error_message', 'Unknown error')}")
        return {'error': result.get('error_message', 'Unknown error')}
    
    points_3d = result['points_3d']
    
    print(f"[+] Triangulated {result['num_points']} 3D points from {result['num_views']} views")
    print(f"   Mean reprojection error: {result['mean_reprojection_error']:.3f} pixels")
    
    # Print per-view errors
    print("\n   Per-view reprojection errors:")
    for err_info in result['reprojection_errors']:
        print(f"     View {err_info['view_index']}: "
              f"mean={err_info['mean_error']:.3f}px, "
              f"max={err_info['max_error']:.3f}px, "
              f"std={err_info['std_error']:.3f}px")
    
    # ========================================
    # STEP 3: QUALITY ANALYSIS
    # ========================================
    
    print("\n[3] Step 3: Analyzing triangulation quality...")
    print("-" * 80)
    
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
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "=" * 80)
    print("EXAMPLE SUMMARY")
    print("=" * 80)
    
    print("\nQuality Metrics:")
    print(f"  Reprojection error: {result['mean_reprojection_error']:.3f} px")
    print(f"  Planarity error: {mean_planarity_error*1000:.3f} mm")
    print(f"  Triangulation angle: {mean_angle:.1f}°")
    
    # ========================================
    # VISUALIZATION
    # ========================================
    
    # Prepare return result
    result_dict = {
        'num_points': result['num_points'],
        'num_views': result['num_views'],
        'mean_reprojection_error': result['mean_reprojection_error'],
        'mean_planarity_error': float(mean_planarity_error),
        'mean_triangulation_angle': float(mean_angle),
        'points_3d': points_3d,
        'camera_centers': result['camera_centers']
    }
    
    if visualize:
        visualize_triangulation_3d(view_data, result_dict)
    
    return {
        'num_points': result['num_points'],
        'num_views': result['num_views'],
        'mean_reprojection_error': result['mean_reprojection_error'],
        'mean_planarity_error': float(mean_planarity_error),
        'mean_triangulation_angle': float(mean_angle),
        'points_3d': points_3d,
        'camera_centers': result['camera_centers']
    }


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
    
    # Run triangulation example
    try:
        print("\n" + "=" * 80)
        print("Running Triangulation Example")
        print("=" * 80)
        
        example_result = run_triangulation_example_with_data(
            view_data=view_data,
            visualize=args.visualize
        )
        
        if 'error' in example_result:
            print(f"\n[!] Example completed with issues: {example_result['error']}")
            sys.exit(1)
        
        print("\n[+] Example completed successfully!")
        print("\nKey Results:")
        print(f"  - Triangulated {example_result['num_points']} 3D points")
        print(f"  - Used {example_result['num_views']} camera views")
        print(f"  - Reprojection error: {example_result['mean_reprojection_error']:.3f} pixels")
        print(f"  - Planarity error: {example_result['mean_planarity_error']*1000:.3f} mm")
        print(f"  - Mean triangulation angle: {example_result['mean_triangulation_angle']:.1f}°")
        sys.exit(0)
    except Exception as e:
        print(f"\n[-] Example failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
