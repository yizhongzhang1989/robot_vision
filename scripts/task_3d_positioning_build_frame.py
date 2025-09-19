#!/usr/bin/env python3
"""
3D Positioning Coordinate System Establishment

This script creates a new coordinate system based on estimated 3D keypoints.
The new coordinate system is defined as:
- Origin: Midpoint between keypoint 2 and keypoint 8
- X-axis: Direction from keypoint 2 to keypoint 8
- Z-axis: Same as base coordinate system Z-axis [0, 0, 1]
- Y-axis: Determined by right-hand rule (Y = Z × X)

Input:
- 3D positioning results from 3d_positioning_estimate.py
- Located at: test_data/output/3d_positioning_results.json

Output:
- New coordinate system transformation matrix
- All keypoints coordinates in new coordinate system
- Results saved to: test_data/output/coordinate_system_results.json
"""

import json
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoordinateSystemBuilder:
    """Class to build new coordinate system based on keypoints"""
    
    def __init__(self, data_dir: str = "test_data"):
        """
        Initialize coordinate system builder
        
        Args:
            data_dir: Base directory containing input and output data
        """
        self.data_dir = Path(data_dir)
        self.results_file = self.data_dir / "output" / "3d_positioning_results.json"
        self.keypoints_3d = None
        self.transformation_matrix = None
        self.origin = None
        
        logger.info(f"Initialized coordinate system builder with data directory: {data_dir}")
    
    def load_3d_positioning_results(self) -> Dict:
        """
        Load 3D positioning estimation results
        
        Returns:
            Dictionary containing 3D positioning results
        """
        try:
            if not self.results_file.exists():
                raise FileNotFoundError(f"3D positioning results file not found: {self.results_file}")
            
            with open(self.results_file, 'r') as f:
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
            raise ValueError("3D positioning results not loaded")
        
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
            raise ValueError("3D positioning results not loaded")
        
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
    
    def create_coordinate_system_results(self, point2_id: int = 2, point8_id: int = 8, point1_id: int = 1) -> Dict:
        """
        Create complete coordinate system results
        
        Args:
            point2_id: ID of the first reference point (origin calculation)
            point8_id: ID of the second reference point (X-axis direction)
            point1_id: ID of the third reference point (kept for compatibility, not used for Z-axis)
            
        Returns:
            Dictionary containing complete results
        """
        # Load 3D positioning results
        positioning_results = self.load_3d_positioning_results()
        
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
    
    def save_results(self, results: Dict, output_file: str = "coordinate_system_results.json"):
        """
        Save coordinate system results to file
        
        Args:
            results: Results dictionary
            output_file: Output filename
        """
        output_path = self.data_dir / "output" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Coordinate system results saved to: {output_path}")
        
        # Print summary
        self.print_results_summary(results)
    
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
    """Main function to run coordinate system establishment"""
    try:
        # Initialize coordinate system builder
        builder = CoordinateSystemBuilder()
        
        # Create coordinate system based on points 2, 8, and 1
        results = builder.create_coordinate_system_results(point2_id=2, point8_id=8, point1_id=1)
        
        # Save results
        builder.save_results(results, "coordinate_system_results.json")
        
        print("\nCoordinate system establishment completed successfully!")
        
    except Exception as e:
        logger.error(f"Coordinate system establishment failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
