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


def run_triangulation_example(visualize=True):
    """
    Run triangulation example using chessboard calibration data.
    
    This function demonstrates:
    1. Intrinsic calibration using chessboard images
    2. Extracting camera poses from calibration
    3. Multi-view triangulation of 3D points
    4. Quality analysis (reprojection error, planarity, spacing)
    5. Optional 3D visualization
    
    Args:
        visualize: If True, display 3D visualization of triangulated points and cameras
    
    Returns:
        Dict containing test results
    """
    print("=" * 80)
    print("Triangulation Example with Chessboard Calibration Data")
    print("=" * 80)
    
    # Add camera calibration toolkit to path
    toolkit_path = parent_dir / "ThirdParty" / "camera_calibration_toolkit"
    
    # Need to temporarily manipulate sys.path and modules to avoid conflict with robot_vision's core/
    original_path = sys.path.copy()
    sys.path.insert(0, str(toolkit_path))
    
    # Import calibration toolkit modules
    try:
        # We need to temporarily remove 'core' from sys.modules to allow import from toolkit's core/
        # Save reference to our triangulation core module
        triangulation_core = sys.modules.get('core')
        triangulation_core_triangulation = sys.modules.get('core.triangulation')
        
        # Temporarily remove core from modules
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
        print(f"❌ Failed to import calibration toolkit: {e}")
        print("\n💡 The camera calibration toolkit submodule may not be initialized.")
        print("   Please run the following commands to update the submodule:")
        print("   git submodule update --init --recursive")
        print(f"\n   Or check if the toolkit exists at: {toolkit_path}")
        sys.path = original_path
        return {'success': False, 'error': 'Calibration toolkit not available'}
    
    # ========================================
    # STEP 1: INTRINSIC CALIBRATION
    # ========================================
    
    print("\n📷 Step 1: Performing intrinsic calibration...")
    print("-" * 80)
    
    sample_dir = toolkit_path / "sample_data" / "eye_in_hand_test_data"
    image_paths = sorted(glob.glob(str(sample_dir / "*.jpg")))
    
    print(f"Found {len(image_paths)} calibration images")
    
    # Load pattern configuration
    config_path = sample_dir / "chessboard_config.json"
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    pattern = load_pattern_from_json(config_data)
    
    print(f"Chessboard pattern: {pattern.width}x{pattern.height}, square size: {pattern.square_size*1000:.1f} mm")
        
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
    rms_error = calib_result.get('rms', calib_result.get('rms_error', 'N/A'))
    
    print(f"✅ Calibration complete (RMS error: {rms_error if isinstance(rms_error, str) else f'{rms_error:.3f}'} pixels)")
    print(f"\nCamera matrix (K):")
    print(camera_matrix)
    print(f"\nDistortion coefficients:")
    print(distortion_coeffs)
        
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
    
    print(f"Valid views: {len(valid_indices)} out of {len(image_points_all)}")
    
    # Use all valid views for triangulation
    test_indices = valid_indices
    
    print(f"Using {len(test_indices)} views for triangulation: indices {test_indices}")
    
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
    print(f"   - Points per view: {len(view_data[0]['points_2d'])}")
    print(f"   - Image size: {view_data[0]['image_size']}")
    
    # ========================================
    # STEP 3: TRIANGULATE 3D POINTS
    # ========================================
    
    print("\n📐 Step 3: Triangulating 3D points...")
    print("-" * 80)
    
    try:
        result = triangulate_multiview(view_data)
    except Exception as e:
        print(f"❌ Exception in triangulate_multiview(): {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    if not result['success']:
        print(f"❌ Triangulation failed: {result.get('error_message', 'Unknown error')}")
        return result
    
    points_3d = result['points_3d']
    
    print(f"✅ Triangulated {result['num_points']} 3D points from {result['num_views']} views")
    print(f"   Mean reprojection error: {result['mean_reprojection_error']:.3f} pixels")
    
    # Print per-view errors
    print("\n   Per-view reprojection errors:")
    for err_info in result['reprojection_errors']:
        print(f"     View {err_info['view_index']}: "
              f"mean={err_info['mean_error']:.3f}px, "
              f"max={err_info['max_error']:.3f}px, "
              f"std={err_info['std_error']:.3f}px")
    
    # ========================================
    # STEP 4: QUALITY ANALYSIS
    # ========================================
    
    print("\n📊 Step 4: Analyzing triangulation quality...")
    print("-" * 80)
    
    # Generate ideal 3D positions of chessboard corners
    objp = np.zeros((pattern.width * pattern.height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern.width, 0:pattern.height].T.reshape(-1, 2)
    objp *= pattern.square_size
    
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
    print("EXAMPLE SUMMARY")
    print("=" * 80)
    
    success = (
        result['mean_reprojection_error'] < 2.0 and
        mean_planarity_error < 0.005 and  # 5mm
        mean_spacing_error < 0.002  # 2mm
    )
    
    if success:
        print("✅ EXCELLENT RESULTS")
    else:
        print("⚠️  ACCEPTABLE RESULTS WITH WARNINGS")
    
    print("\nQuality Metrics:")
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
            import matplotlib
            matplotlib.use('TkAgg')  # Use interactive backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot triangulated 3D points
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                      c='blue', marker='o', s=30, label='Triangulated Points', alpha=0.8)
            
            # Plot camera positions
            camera_centers = result['camera_centers']
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
            
            # Draw chessboard grid lines
            for i in range(pattern.height):
                row_points = points_3d[i*pattern.width:(i+1)*pattern.width]
                ax.plot(row_points[:, 0], row_points[:, 1], row_points[:, 2], 
                       'g-', alpha=0.6, linewidth=1)
            
            for j in range(pattern.width):
                col_points = points_3d[j::pattern.width]
                ax.plot(col_points[:, 0], col_points[:, 1], col_points[:, 2], 
                       'g-', alpha=0.6, linewidth=1)
            
            # Set labels and title
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            ax.set_title(f'3D Triangulation Result\n{result["num_points"]} points from {result["num_views"]} views\n'
                        f'Mean reprojection error: {result["mean_reprojection_error"]:.3f} px',
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
            
            print("✅ Visualization closed")
            
        except ImportError as e:
            print(f"⚠️  Could not create visualization: matplotlib not available ({e})")
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    return {
        'success': success,
        'num_points': result['num_points'],
        'num_views': result['num_views'],
        'mean_reprojection_error': result['mean_reprojection_error'],
        'mean_planarity_error': float(mean_planarity_error),
        'mean_spacing_error': float(mean_spacing_error),
        'mean_triangulation_angle': float(mean_angle),
        'points_3d': points_3d,
        'camera_centers': result['camera_centers']
    }


if __name__ == "__main__":
    print("Multi-View Triangulation Example")
    print("=" * 80)
    print("\nThis example demonstrates how to triangulate 3D points from multiple views")
    print("using chessboard calibration data.")
    print("=" * 80)
    
    try:
        # Run example with visualization enabled
        example_result = run_triangulation_example(visualize=True)
        
        if example_result['success']:
            print("\n🎉 Example completed successfully!")
            print(f"\nKey Results:")
            print(f"  - Triangulated {example_result['num_points']} 3D points")
            print(f"  - Used {example_result['num_views']} camera views")
            print(f"  - Reprojection error: {example_result['mean_reprojection_error']:.3f} pixels")
            print(f"  - Planarity error: {example_result['mean_planarity_error']*1000:.3f} mm")
            print(f"  - Spacing error: {example_result['mean_spacing_error']*1000:.3f} mm")
            sys.exit(0)
        else:
            print(f"\n⚠️  Example completed with issues: {example_result.get('error', 'See details above')}")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Example failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
