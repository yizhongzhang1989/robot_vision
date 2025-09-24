#!/usr/bin/env python3
"""
Cabinet Frame Coordinate System Establishment

This script creates a new coordinate system based on 3D keypoint positioning results.
It loads pre-computed 3D keypoint coordinates and builds a cabinet-specific coordinate system.

The new coordinate system is defined as:
- Origin: Midpoint between keypoint 2 and keypoint 8
- X-axis: Direction from keypoint 2 to keypoint 8, projected onto XOY plane (horizontal)
- Z-axis: Same as base coordinate system Z-axis [0, 0, 1] (vertical)
- Y-axis: Determined by right-hand rule (Y = Z × X)

Note: The X-axis projection ensures that the cabinet coordinate system is always
horizontal, which is appropriate for cabinet-like objects that should have a
level orientation.

Input:
- 3D positioning results from ffpp_positioning_results.json

Output:
- New coordinate system transformation matrix
- Cabinet pose information (position and orientation)
- Results saved to: temp/cabinet_frame_result/cabinet_frame_results.json
"""

import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add project paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# No additional imports needed for current functionality

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CabinetFrameBuilder:
    """Class to build cabinet coordinate system from 3D keypoint positioning results"""
    
    def __init__(self, data_dir: str = "test_data"):
        """
        Initialize cabinet frame builder
        
        Args:
            data_dir: Base directory containing input and output data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("temp") / "cabinet_frame_result"  # Specific output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to 3D positioning results
        self.positioning_results_file = Path("temp") / "3d_coordinate_estimation_result" / "3d_coordinates_estimation_result.json"
        
        # No tracker needed for current functionality
        
        # State variables
        self.keypoints_3d = None
        self.transformation_matrix = None
        self.origin = None
        
        logger.info(f"Initialized cabinet frame builder with data directory: {data_dir}")
        logger.info(f"3D positioning results file: {self.positioning_results_file}")
    
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
    
    def build_coordinate_system(self, point2_id: int = 2, point8_id: int = 8) -> np.ndarray:
        """
        Build new coordinate system based on keypoints
        
        The new coordinate system is defined as:
        - Origin: Midpoint between point2 and point8
        - X-axis: Direction from point2 to point8, projected onto XOY plane (horizontal)
        - Z-axis: Same as base coordinate system Z-axis [0, 0, 1] (vertical)
        - Y-axis: Cross product using right-hand rule (Y = Z × X)
        
        Note: The X-axis projection ensures that the cabinet coordinate system
        is always horizontal, which is appropriate for cabinet-like objects.
        
        Args:
            point2_id: ID of the first reference point (default: 2)
            point8_id: ID of the second reference point for X-axis (default: 8)
            
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
        # First get the raw direction vector
        x_direction_raw = p8_coords - p2_coords
        x_axis_raw = x_direction_raw / np.linalg.norm(x_direction_raw)
        logger.info(f"Raw X-axis direction (calculated from {point2_id} to {point8_id}): {x_axis_raw}")

        # Project to XOY plane to ensure cabinet X-axis is horizontal
        x_direction_projected = np.array([x_direction_raw[0], x_direction_raw[1], 0.0])
        
        # Check for degenerate case (if projection is too small)
        if np.linalg.norm(x_direction_projected) < 1e-6:
            logger.warning("X-axis direction is nearly vertical, using base X-axis as default")
            x_direction_projected = np.array([1.0, 0.0, 0.0])
        
        # Normalize to get unit X-axis
        x_axis = x_direction_projected / np.linalg.norm(x_direction_projected)
        logger.info(f"Adjusted X-axis direction (parallel to XOY plane): {x_axis}")
        logger.info(f"Ideal cabinet X+ = Base Y+, reference value =  [0.00000000, 1.00000000, 0.00000000]")

        
        # Use base coordinate system Z-axis
        z_axis = np.array([0.0, 0.0, 1.0])
        logger.info(f"Z-axis direction (same as base Z-axis): {z_axis}")
        logger.info(f"Ideal cabinet Z+ = Base Z+, reference value =  [0.00000000, 0.00000000, 1.00000000]")

        # Y-axis is calculated using right-hand rule: Y = Z × X
        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = np.linalg.norm(y_axis)
        
        if y_axis_norm < 1e-6:
            raise ValueError("X-axis and Z-axis are parallel, cannot create coordinate system")
        
        y_axis = y_axis / y_axis_norm
        logger.info(f"Y-axis direction (calculated from Y = Z × X): {y_axis}")
        logger.info(f"Ideal cabinet Y+ = Base X-, reference value =  [-1.00000000, 0.00000000, 0.00000000]")

        # Verify the axes form an orthonormal system
        rotation_matrix_check = np.column_stack([x_axis, y_axis, z_axis])
        logger.info(f"Orthonormal check - det(R): {np.linalg.det(rotation_matrix_check):.6f} (should be 1.0)")
        
        # Verify orthogonality
        dot_xy = np.dot(x_axis, y_axis)
        dot_xz = np.dot(x_axis, z_axis)
        dot_yz = np.dot(y_axis, z_axis)
        
        logger.info(f"Orthogonality check - X·Y: {dot_xy:.6f}, X·Z: {dot_xz:.6f}, Y·Z: {dot_yz:.6f} (all should be 0.0)")
        
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

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion (w, x, y, z)
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion as numpy array [w, x, y, z]
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
            
        return np.array([qw, qx, qy, qz])

    def quaternion_angle_difference(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Calculate the angular difference between two quaternions
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            
        Returns:
            Angular difference in degrees
        """
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Calculate dot product
        dot_product = np.abs(np.dot(q1, q2))
        
        # Clamp to avoid numerical errors
        dot_product = np.clip(dot_product, 0.0, 1.0)
        
        # Calculate angle difference (in radians, then convert to degrees)
        angle_rad = 2 * np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

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
        
        # Here note that the "columns" of the rotation matrix represent the cabinet axes in base frame
        ideal_rotation_matrix = np.array([
            [0, -1,  0],   # Cabinet X+ = Base Y+ = [0, 1, 0]
            [1,  0,  0],   # Cabinet Y+ = Base X- = [-1, 0, 0]
            [0,  0,  1]    # Cabinet Z+ = Base Z+ = [0, 0, 1]
        ])
        
        # Calculate Euler angles from ideal rotation matrix
        rx, ry, rz = self.rotation_matrix_to_euler_angles(ideal_rotation_matrix)
        
        return rx, ry, rz, ideal_rotation_matrix
    
    def create_cabinet_frame_results(self, point2_id: int = 2, point8_id: int = 8) -> Dict:
        """
        Create complete cabinet frame results
        
        Args:
            point2_id: ID of the first reference point (origin calculation)
            point8_id: ID of the second reference point (X-axis direction)
            
        Returns:
            Dictionary containing complete results
        """
        # Build coordinate system
        transformation_matrix = self.build_coordinate_system(point2_id, point8_id)
        
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
        
        # Calculate quaternion-based deviation
        estimated_quaternion = self.rotation_matrix_to_quaternion(estimated_rotation_matrix)
        ideal_quaternion = self.rotation_matrix_to_quaternion(ideal_rotation_matrix)
        angular_deviation = self.quaternion_angle_difference(estimated_quaternion, ideal_quaternion)
        
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
            'estimated_cabinet_quaternion': [
                float(estimated_quaternion[0]),
                float(estimated_quaternion[1]),
                float(estimated_quaternion[2]),
                float(estimated_quaternion[3])
            ],
            'ideal_cabinet_quaternion': [
                float(ideal_quaternion[0]),
                float(ideal_quaternion[1]),
                float(ideal_quaternion[2]),
                float(ideal_quaternion[3])
            ],
            'angular_deviation_degrees': float(angular_deviation),
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
        
        print("\n📊 Cabinet Frame Results Summary:")
        cabinet2base_matrix = np.array(results['estimated_cabinet2base_matrix'])
        rotation_matrix = cabinet2base_matrix[:3, :3]
        
        # Get origin coordinates
        origin = results['estimated_cabinet_xyz_in_base']
        print("Cabinet Frame Origin in Base Frame:")
        print(f"  [{origin[0]:.6f}, {origin[1]:.6f}, {origin[2]:.6f}] m")

        print("\nCabinet2Base Transformation Matrix:")
        for i, row in enumerate(cabinet2base_matrix):
            print(f"  [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}, {row[3]:9.6f}]")
        
        # Calculate and display ideal cabinet rotation angles
        ideal_rx, ideal_ry, ideal_rz, _ = self.calculate_ideal_cabinet_rotation()
        print("\nIdeal Cabinet Frame Pose Relative to Base Frame:")
        print(f"  X-axis rotation: {ideal_rx:.3f}° (Roll)")
        print(f"  Y-axis rotation: {ideal_ry:.3f}° (Pitch)")
        print(f"  Z-axis rotation: {ideal_rz:.3f}° (Yaw)")
        print(" (Ideal Cabinet X+ = Base Y+, Y+ = Base X-, Z+ = Base Z+)")
        
        # Calculate rotation angles for actual cabinet axes relative to base frame
        rx, ry, rz = self.rotation_matrix_to_euler_angles(rotation_matrix)
        print("\nActual Cabinet Frame Pose Relative to Base Frame:")
        print(f"  X-axis rotation: {rx:.3f}° (Roll)")
        print(f"  Y-axis rotation: {ry:.3f}° (Pitch)")
        print(f"  Z-axis rotation: {rz:.3f}° (Yaw)")
        
        # Calculate deviation from ideal using quaternions (more reliable than Euler angles)
        estimated_quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
        _, _, _, ideal_rotation_matrix = self.calculate_ideal_cabinet_rotation()
        ideal_quaternion = self.rotation_matrix_to_quaternion(ideal_rotation_matrix)
        deviation_angle = self.quaternion_angle_difference(estimated_quaternion, ideal_quaternion)
        
        print("\nDeviation from Ideal Cabinet Orientation:")
        print(f"    Roll deviation:  {rx - ideal_rx:+.3f}°")
        print(f"    Pitch deviation: {ry - ideal_ry:+.3f}°")
        print(f"    Yaw deviation:   {rz - ideal_rz:+.3f}°")


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
        
        print("🏗️  Building Cabinet Frame Coordinate System")
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
                
        results = builder.create_cabinet_frame_results(point2_id=point2_id, point8_id=point8_id)
        
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