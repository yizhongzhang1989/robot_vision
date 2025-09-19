#!/usr/bin/env python3
"""
Cabinet Frame Coordinate System Establishment

This script creates a new coordinate system based on tracked keypoints using FFPPKeypointTracker.
Similar to task_3d_positioning_build_frame.py but uses the FFPPKeypointTracker class for
improved keypoint tracking performance.

The new coordinate system is defined as:
- Origin: Midpoint between keypoint 2 and keypoint 8
- X-axis: Direction from keypoint 2 to keypoint 8
- Z-axis: Same as base coordinate system Z-axis [0, 0, 1]
- Y-axis: Determined by right-hand rule (Y = Z × X)

Input:
- Reference image with keypoints
- Target images for keypoint tracking
- FFPPKeypointTracker for efficient tracking

Output:
- New coordinate system transformation matrix
- All keypoints coordinates in new coordinate system
- Results saved to: test_data/output/cabinet_frame_results.json
"""

import json
import numpy as np
import os
import sys
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image

# Add project paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import FFPPKeypointTracker and utilities
from core.ffpp_keypoint_tracker import FFPPKeypointTracker
from core.utils import load_keypoints, visualize_tracking_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CabinetFrameBuilder:
    """Class to build cabinet coordinate system using FFPPKeypointTracker"""
    
    def __init__(self, data_dir: str = "test_data"):
        """
        Initialize cabinet frame builder
        
        Args:
            data_dir: Base directory containing input and output data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("output") / "cabinet_frame_build_output"  # Specific output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to 3D positioning results
        self.positioning_results_file = Path("output") / "pose_estimate_output" / "ffpp_positioning_results.json"
        
        # Initialize FFPPKeypointTracker (optional, for additional tracking if needed)
        self.tracker = FFPPKeypointTracker()
        
        # State variables
        self.keypoints_3d = None
        self.transformation_matrix = None
        self.origin = None
        
        logger.info(f"Initialized cabinet frame builder with data directory: {data_dir}")
        logger.info(f"3D positioning results file: {self.positioning_results_file}")
    
    def load_reference_image_and_keypoints(self, 
                                         ref_image_path: str, 
                                         keypoints_path: str) -> Dict:
        """
        Load reference image and keypoints, set up tracker
        
        Args:
            ref_image_path: Path to reference image
            keypoints_path: Path to keypoints JSON file
            
        Returns:
            Dictionary containing loading results
        """
        try:
            # Load reference image
            ref_img_pil = Image.open(ref_image_path)
            ref_img = np.array(ref_img_pil)
            
            # Load keypoints
            keypoints, original_size = load_keypoints(keypoints_path)
            
            logger.info(f"Loaded reference image: {ref_image_path}")
            logger.info(f"Image size: {original_size}")
            logger.info(f"Loaded {len(keypoints)} keypoints from {keypoints_path}")
            
            # Set reference image in tracker
            result = self.tracker.set_reference_image(ref_img, keypoints)
            
            if not result['success']:
                raise ValueError(f"Failed to set reference image: {result['error']}")
            
            return {
                'success': True,
                'reference_image': ref_img,
                'keypoints': keypoints,
                'original_size': original_size,
                'tracker_result': result
            }
            
        except Exception as e:
            logger.error(f"Failed to load reference image and keypoints: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def load_3d_positioning_results(self) -> Dict:
        """
        Load 3D positioning results from ffpp_positioning_results.json
        
        Returns:
            Dictionary containing 3D positioning results
        """
        try:
            if not self.positioning_results_file.exists():
                raise FileNotFoundError(f"3D positioning results file not found: {self.positioning_results_file}")
            
            with open(self.positioning_results_file, 'r') as f:
                results = json.load(f)
            
            if not results.get('success', False):
                raise ValueError("3D positioning estimation was not successful")
            
            self.keypoints_3d = results['keypoints_3d']
            logger.info(f"Loaded 3D positioning results with {len(self.keypoints_3d)} keypoints")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to load 3D positioning results: {e}")
            raise
    
    def track_keypoints_in_target_images(self, target_images: List[str]) -> Dict:
        """
        Track keypoints in multiple target images using FFPPKeypointTracker
        This method is optional and can be used for additional validation
        
        Args:
            target_images: List of paths to target images
            
        Returns:
            Dictionary containing tracking results for all images
        """
        all_tracking_results = []
        
        try:
            for i, target_path in enumerate(target_images):
                logger.info(f"Processing target image {i+1}/{len(target_images)}: {target_path}")
                
                # Load target image
                target_img_pil = Image.open(target_path)
                target_img = np.array(target_img_pil)
                
                # Track keypoints
                tracking_result = self.tracker.track_keypoints(target_img, bidirectional=True)
                
                if not tracking_result['success']:
                    logger.error(f"Failed to track keypoints in {target_path}: {tracking_result['error']}")
                    continue
                
                # Store result with image path
                result_with_path = tracking_result.copy()
                result_with_path['image_path'] = target_path
                result_with_path['image_index'] = i
                
                all_tracking_results.append(result_with_path)
                
                logger.info(f"Successfully tracked {len(tracking_result['tracked_keypoints'])} keypoints")
            
            return {
                'success': True,
                'tracking_results': all_tracking_results,
                'processed_images': len(all_tracking_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to track keypoints: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def find_keypoint_by_id(self, keypoint_id: int) -> Optional[Dict]:
        """
        Find keypoint by ID
        
        Args:
            keypoint_id: ID of the keypoint to find
            
        Returns:
            Keypoint dictionary or None if not found
        """
        if self.keypoints_3d is None:
            raise ValueError("3D keypoints not available")
        
        for kp in self.keypoints_3d:
            if kp['id'] == keypoint_id:
                return kp
        
        return None
    
    def extract_3d_coordinates(self, keypoint: Dict) -> np.ndarray:
        """
        Extract 3D coordinates from keypoint dictionary
        
        Args:
            keypoint: Keypoint dictionary containing coordinates_3d
            
        Returns:
            3D coordinates as numpy array [x, y, z]
        """
        coords = keypoint['coordinates_3d']
        return np.array([coords['x'], coords['y'], coords['z']])
    
    def build_coordinate_system(self, point2_id: int = 2, point8_id: int = 8, point1_id: int = 1) -> np.ndarray:
        """
        Build new coordinate system based on keypoints
        
        The new coordinate system is defined as:
        - Origin: Midpoint between point2 and point8
        - X-axis: Direction from point2 to point8 (normalized)
        - Z-axis: Same as base coordinate system Z-axis [0, 0, 1]
        - Y-axis: Cross product using right-hand rule (Y = Z × X)
        
        Args:
            point2_id: ID of the first reference point (default: 2)
            point8_id: ID of the second reference point for X-axis (default: 8)
            point1_id: ID of the third reference point (not used for Z-axis anymore, kept for compatibility)
            
        Returns:
            4x4 transformation matrix from base to new coordinate system
        """
        if self.keypoints_3d is None:
            raise ValueError("3D keypoints not available")
        
        # Find the reference keypoints
        point2 = self.find_keypoint_by_id(point2_id)
        point8 = self.find_keypoint_by_id(point8_id)
        
        if point2 is None:
            raise ValueError(f"Keypoint with ID {point2_id} not found")
        if point8 is None:
            raise ValueError(f"Keypoint with ID {point8_id} not found")
        
        # Extract 3D coordinates
        p2_coords = self.extract_3d_coordinates(point2)
        p8_coords = self.extract_3d_coordinates(point8)
        
        logger.info(f"Point {point2_id} coordinates: {p2_coords}")
        logger.info(f"Point {point8_id} coordinates: {p8_coords}")
        
        # Calculate origin (midpoint between point2 and point8)
        self.origin = (p2_coords + p8_coords) / 2.0
        logger.info(f"New coordinate system origin: {self.origin}")
        
        # Calculate X-axis (direction from point2 to point8)
        x_direction = p8_coords - p2_coords
        x_axis = x_direction / np.linalg.norm(x_direction)
        logger.info(f"X-axis direction (point {point2_id} to {point8_id}): {x_axis}")
        
        # Use base coordinate system Z-axis
        z_axis = np.array([0.0, 0.0, 1.0])
        logger.info(f"Z-axis direction (same as base Z-axis): {z_axis}")
        
        # Y-axis is calculated using right-hand rule: Y = Z × X
        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = np.linalg.norm(y_axis)
        
        if y_axis_norm < 1e-6:
            raise ValueError("X-axis and Z-axis are parallel, cannot create coordinate system")
        
        y_axis = y_axis / y_axis_norm
        logger.info(f"Y-axis direction (Z × X): {y_axis}")
        
        # Verify orthogonality
        dot_xy = np.dot(x_axis, y_axis)
        dot_xz = np.dot(x_axis, z_axis)
        dot_yz = np.dot(y_axis, z_axis)
        
        logger.info(f"Orthogonality check - X·Y: {dot_xy:.6f}, X·Z: {dot_xz:.6f}, Y·Z: {dot_yz:.6f}")
        
        # Check if axes are reasonably orthogonal (allow some tolerance for measurement errors)
        max_dot_product = max(abs(dot_xy), abs(dot_xz), abs(dot_yz))
        if max_dot_product > 0.1:  # ~5.7 degrees
            logger.warning(f"Coordinate system axes have significant non-orthogonality: max dot product = {max_dot_product:.6f}")
        elif max_dot_product > 1e-3:  # ~0.06 degrees
            logger.info(f"Coordinate system axes have minor non-orthogonality: max dot product = {max_dot_product:.6f}")
        
        # Create transformation matrix
        # This matrix transforms points from base coordinates to new coordinates
        # T_new_from_base = [R | -R*t; 0 0 0 1]
        # where R is rotation matrix and t is translation (origin)
        
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3, :3] = rotation_matrix.T  # Transpose for base-to-new transformation
        self.transformation_matrix[:3, 3] = -rotation_matrix.T @ self.origin
        
        logger.info("Coordinate system transformation matrix created successfully")
        
        return self.transformation_matrix
    
    def transform_point_to_new_coordinate_system(self, point_base: np.ndarray) -> np.ndarray:
        """
        Transform a point from base coordinate system to new coordinate system
        
        Args:
            point_base: 3D point in base coordinate system
            
        Returns:
            3D point in new coordinate system
        """
        if self.transformation_matrix is None:
            raise ValueError("Coordinate system not built yet")
        
        # Convert to homogeneous coordinates
        point_homo = np.append(point_base, 1.0)
        
        # Apply transformation
        point_new_homo = self.transformation_matrix @ point_homo
        
        # Return 3D coordinates
        return point_new_homo[:3]
    
    def transform_all_keypoints(self) -> List[Dict]:
        """
        Transform all keypoints to new coordinate system
        
        Returns:
            List of keypoints with coordinates in new coordinate system
        """
        if self.keypoints_3d is None or self.transformation_matrix is None:
            raise ValueError("3D positioning results or coordinate system not available")
        
        transformed_keypoints = []
        
        for kp in self.keypoints_3d:
            # Extract original coordinates
            original_coords = self.extract_3d_coordinates(kp)
            
            # Transform to new coordinate system
            new_coords = self.transform_point_to_new_coordinate_system(original_coords)
            
            # Create new keypoint entry
            transformed_kp = {
                'id': kp['id'],
                'name': kp['name'],
                'coordinates_base': {
                    'x': float(original_coords[0]),
                    'y': float(original_coords[1]),
                    'z': float(original_coords[2])
                },
                'coordinates_new': {
                    'x': float(new_coords[0]),
                    'y': float(new_coords[1]),
                    'z': float(new_coords[2])
                },
                'reference_2d': kp['reference_2d']
            }
            
            transformed_keypoints.append(transformed_kp)
        
        logger.info(f"Transformed {len(transformed_keypoints)} keypoints to new coordinate system")
        
        return transformed_keypoints
    
    def rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (ZYX intrinsic convention)
        
        This extracts Euler angles from rotation matrix using ZYX convention.
        ZYX intrinsic (or ZYX extrinsic) means: Z-axis rotation, then Y-axis, then X-axis.
        This corresponds to: Yaw (rz), then Pitch (ry), then Roll (rx).
        
        Args:
            R: 3x3 rotation matrix (from R = Rz * Ry * Rx)
            
        Returns:
            Tuple of (rx, ry, rz) in degrees
        """
        # Extract Euler angles from rotation matrix (ZYX convention)
        # This implementation extracts angles from R = Rz * Ry * Rx
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            rx = np.arctan2(R[2,1], R[2,2])  # Roll (X-axis rotation)
            ry = np.arctan2(-R[2,0], sy)     # Pitch (Y-axis rotation)  
            rz = np.arctan2(R[1,0], R[0,0])  # Yaw (Z-axis rotation)
        else:
            # Gimbal lock case
            rx = np.arctan2(-R[1,2], R[1,1])
            ry = np.arctan2(-R[2,0], sy)
            rz = 0
        
        # Convert from radians to degrees
        return np.degrees(rx), np.degrees(ry), np.degrees(rz)

    def calculate_ideal_cabinet_rotation(self) -> Tuple[float, float, float, np.ndarray]:
        """
        Calculate rotation angles and matrix for ideal cabinet coordinate system relative to base frame
        
        Ideal cabinet coordinate system definition:
        - Cabinet X+ aligns with Base Y+
        - Cabinet Y+ aligns with Base X-
        - Cabinet Z+ aligns with Base Z+
        
        Returns:
            Tuple of (rx, ry, rz, rotation_matrix) where angles are in degrees
        """
        # Define ideal cabinet coordinate system relative to base
        # Cabinet X+ = Base Y+ = [0, 1, 0]
        # Cabinet Y+ = Base X- = [-1, 0, 0]  
        # Cabinet Z+ = Base Z+ = [0, 0, 1]
        
        ideal_rotation_matrix = np.array([
            [0,  1,  0],   # Cabinet X+ = Base Y+
            [-1, 0,  0],   # Cabinet Y+ = Base X-
            [0,  0,  1]    # Cabinet Z+ = Base Z+
        ])
        
        # Calculate Euler angles from ideal rotation matrix
        rx, ry, rz = self.rotation_matrix_to_euler_angles(ideal_rotation_matrix)
        
        return rx, ry, rz, ideal_rotation_matrix
    
    def create_cabinet_frame_results(self, point2_id: int = 2, point8_id: int = 8, point1_id: int = 1) -> Dict:
        """
        Create complete cabinet frame results
        
        Args:
            point2_id: ID of the first reference point (origin calculation)
            point8_id: ID of the second reference point (X-axis direction)
            point1_id: ID of the third reference point (kept for compatibility, not used for Z-axis)
            
        Returns:
            Dictionary containing complete results
        """
        # Build coordinate system
        transformation_matrix = self.build_coordinate_system(point2_id, point8_id, point1_id)
        
        # Transform all keypoints
        transformed_keypoints = self.transform_all_keypoints()
        
        # Calculate distances between reference points
        point2 = self.find_keypoint_by_id(point2_id)
        point8 = self.find_keypoint_by_id(point8_id)
        p2_coords = self.extract_3d_coordinates(point2)
        p8_coords = self.extract_3d_coordinates(point8)
        
        x_axis_distance = np.linalg.norm(p8_coords - p2_coords)  # Distance for X-axis
        
        # Calculate cabinet2base transformation matrix (inverse of base2cabinet)
        cabinet2base_matrix = np.linalg.inv(transformation_matrix)
        
        # Calculate rotation angles for estimated cabinet
        estimated_rotation_matrix = cabinet2base_matrix[:3, :3]
        estimated_rx, estimated_ry, estimated_rz = self.rotation_matrix_to_euler_angles(estimated_rotation_matrix)
        
        # Calculate ideal cabinet rotation angles and matrix
        ideal_rx, ideal_ry, ideal_rz, ideal_rotation_matrix = self.calculate_ideal_cabinet_rotation()
        
        # Create ideal cabinet transformation matrices
        # Ideal cabinet2base matrix: transforms points from ideal cabinet to base coordinates
        ideal_cabinet2base_matrix = np.eye(4)
        ideal_cabinet2base_matrix[:3, :3] = ideal_rotation_matrix
        ideal_cabinet2base_matrix[:3, 3] = self.origin  # Use same origin as estimated cabinet
        
        # Ideal base2cabinet matrix: transforms points from base to ideal cabinet coordinates
        ideal_base2cabinet_matrix = np.linalg.inv(ideal_cabinet2base_matrix)
        
        # Create results dictionary with requested fields
        results = {
            'success': True,
            'estimated_cabinet_xyz_in_base': [
                float(self.origin[0]),
                float(self.origin[1]), 
                float(self.origin[2])
            ],
            'estimated_cabinet_rpy_in_base': [
                float(estimated_rx),
                float(estimated_ry),
                float(estimated_rz)
            ],
            'ideal_cabinet_rpy_in_base': [
                float(ideal_rx),
                float(ideal_ry),
                float(ideal_rz)
            ],
            'estimated_cabinet2base_matrix': cabinet2base_matrix.tolist(),
            'estimated_base2cabinet_matrix': transformation_matrix.tolist(),
            'ideal_cabinet2base_matrix': ideal_cabinet2base_matrix.tolist(),
            'ideal_base2cabinet_matrix': ideal_base2cabinet_matrix.tolist()
        }
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "cabinet_frame_results.json"):
        """
        Save cabinet frame results to file
        
        Args:
            results: Results dictionary
            output_file: Output filename
        """
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Cabinet frame results saved to: {output_path}")
        
        # Print summary
        self.print_results_summary(results)
    
    def print_results_summary(self, results: Dict):
        """
        Print a simplified summary of the coordinate system results
        
        Args:
            results: Results dictionary
        """
        print("\n" + "="*60)
        print("CABINET COORDINATE SYSTEM RESULTS")
        print("="*60)
        
        cabinet2base_matrix = np.array(results['estimated_cabinet2base_matrix'])
        rotation_matrix = cabinet2base_matrix[:3, :3]
        
        # Get origin coordinates
        origin = results['estimated_cabinet_xyz_in_base']
        print("Cabinet Origin in Base Frame:")
        print(f"  [{origin[0]:.6f}, {origin[1]:.6f}, {origin[2]:.6f}] m")
        
        # Calculate and display ideal cabinet rotation angles
        ideal_rx, ideal_ry, ideal_rz, _ = self.calculate_ideal_cabinet_rotation()
        print("\nIdeal Cabinet Axes Rotation Relative to Base Frame:")
        print(f"  X-axis rotation: {ideal_rx:.3f}° (Roll)   - Cabinet X+ should align with Base Y+")
        print(f"  Y-axis rotation: {ideal_ry:.3f}° (Pitch)  - Cabinet Y+ should align with Base X-")
        print(f"  Z-axis rotation: {ideal_rz:.3f}° (Yaw)    - Cabinet Z+ should align with Base Z+")
        
        # Calculate rotation angles for actual cabinet axes relative to base frame
        rx, ry, rz = self.rotation_matrix_to_euler_angles(rotation_matrix)
        print("\nActual Cabinet Axes Rotation Relative to Base Frame:")
        print(f"  X-axis rotation: {rx:.3f}° (Roll)")
        print(f"  Y-axis rotation: {ry:.3f}° (Pitch)")
        print(f"  Z-axis rotation: {rz:.3f}° (Yaw)")
        
        # Calculate deviation from ideal
        deviation_rx = rx - ideal_rx
        deviation_ry = ry - ideal_ry
        deviation_rz = rz - ideal_rz
        
        print("\nDeviation from Ideal Cabinet Orientation:")
        print(f"  Roll deviation:  {deviation_rx:+.3f}°")
        print(f"  Pitch deviation: {deviation_ry:+.3f}°")
        print(f"  Yaw deviation:   {deviation_rz:+.3f}°")
        
        print("\nCabinet2Base Transformation Matrix:")
        for i, row in enumerate(cabinet2base_matrix):
            print(f"  [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}, {row[3]:9.6f}]")


def main():
    """Main function to run cabinet frame coordinate system establishment"""
    try:
        # Initialize cabinet frame builder
        builder = CabinetFrameBuilder()
        
        # Check if 3D positioning results file exists
        if not builder.positioning_results_file.exists():
            print(f"❌ Missing required 3D positioning results file:")
            print(f"   - {builder.positioning_results_file}")
            print("\nPlease run the 3D positioning script first to generate the results file.")
            return 1
        
        print("🏗️  Building Cabinet Frame Coordinate System with FFPPKeypointTracker")
        print("=" * 70)
        
        # Step 1: Load 3D positioning results
        print("\n📍 Loading 3D positioning results...")
        positioning_data = builder.load_3d_positioning_results()
        
        print(f"✅ Loaded 3D positioning results with {len(builder.keypoints_3d)} keypoints")
        print(f"   Processing {positioning_data['num_views']} views with {positioning_data['num_keypoints']} keypoints")
        
        # Display available keypoints
        print("\n📍 Available keypoints:")
        for kp in builder.keypoints_3d[:5]:  # Show first 5
            coords = kp['coordinates_3d']
            print(f"   - ID {kp['id']}: {kp['name']} at ({coords['x']:.3f}, {coords['y']:.3f}, {coords['z']:.3f})")
        if len(builder.keypoints_3d) > 5:
            print(f"   ... and {len(builder.keypoints_3d) - 5} more keypoints")
        
        # Step 2: Build cabinet coordinate system
        print("\n🏗️  Building cabinet coordinate system...")
        # Use point2 (ID=2) and point8 (ID=8) as in the original script
        available_ids = [kp['id'] for kp in builder.keypoints_3d]
        
        # Check if we have the required keypoints (2 and 8)
        point2_id = 2
        point8_id = 8
        
        if point2_id not in available_ids:
            print(f"   Warning: Point ID {point2_id} not found, using first available point instead")
            point2_id = available_ids[0]
        
        if point8_id not in available_ids:
            print(f"   Warning: Point ID {point8_id} not found, using second available point instead")
            if len(available_ids) > 1:
                point8_id = available_ids[1]
            else:
                raise ValueError(f"Need at least 2 keypoints to build coordinate system, but only {len(available_ids)} available")
        
        print(f"   Using keypoints ID {point2_id} and ID {point8_id} to define X-axis (point {point2_id} → point {point8_id})")
        
        results = builder.create_cabinet_frame_results(point2_id=point2_id, point8_id=point8_id, point1_id=1)
        
        # Step 3: Save results
        print("\n💾 Saving results...")
        builder.save_results(results, "cabinet_frame_results.json")
        
        print("\n🎉 Cabinet frame coordinate system establishment completed successfully!")
        print(f"   Results saved to: {builder.output_dir / 'cabinet_frame_results.json'}")
        
    except Exception as e:
        logger.error(f"Cabinet frame establishment failed: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())