import json
import numpy as np
import os
import time  
from math import radians  
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the robot directory and its lib subdirectory to the Python path
robot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'robot')
sys.path.insert(0, robot_dir)
sys.path.insert(0, os.path.join(robot_dir, 'lib'))

from DucoCobot import DucoCobot  
from gen_py.robot.ttypes import Op  
from thrift import Thrift  
   
# Robot connection parameters  
ip = '192.168.1.10'     # real robot
port = 7003  


def get_start_point_in_base():
    """
    Calculate start point position and orientation in base coordinate system.
    
    Returns:
        tuple: (position_xyz, rpy_angles)
            - position_xyz: [x, y, z] in meters
            - rpy_angles: [rx, ry, rz] in degrees (ZYX intrinsic rpy angles)
    """
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    coord_file = os.path.join(project_root, 'test_data', 'output', 'coordinate_system_results.json')
    pose_file = os.path.join(project_root, 'test_data', 'positioning_data', 'task_start_pose.json')
    
    # Read data
    with open(coord_file, 'r') as f:
        coord_data = json.load(f)
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    
    # Get transformation matrix from cabinet to base
    cabinet2base = np.array(coord_data['estimated_cabinet2base_matrix'])
    
    # Get task pose in cabinet coordinate (position and rotation)
    task_pos = np.array(pose_data['position'])
    task_rot = np.array(pose_data['rotation_matrix'])
    
    # Create 4x4 transformation matrix for task pose in cabinet
    task_matrix = np.eye(4)
    task_matrix[:3, :3] = task_rot
    task_matrix[:3, 3] = task_pos
    
    # Transform to base coordinate: T_base = T_cabinet2base * T_task_cabinet
    result_matrix = cabinet2base @ task_matrix
    
    # Extract position
    xyz = result_matrix[:3, 3].tolist()
    
    # Convert rotation matrix to rpy angles (ZYX intrinsic convention)
    # This extracts rx, ry, rz corresponding to R = Rz * Ry * Rx
    R = result_matrix[:3, :3]
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])   # rx - final rotation around X-axis
        ry = np.arctan2(-R[2, 0], sy)       # ry - middle rotation around Y-axis
        rz = np.arctan2(R[1, 0], R[0, 0])   # rz - first rotation around Z-axis
    else:
        # Gimbal lock case
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0
    
    # Convert to degrees and to float type
    rpy_angles = [float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))]
    
    return xyz, rpy_angles



def calculate_target_joint_angles(robot):
    """
    Calculate target joint angles for the robot from the start point in base coordinates.

    Returns:
    - list of joint angles (radians) on success, or None if inverse kinematics fails.
    """
    target_xyz_m, target_rpy_deg = get_start_point_in_base()

    # target_xyz_mm = [float(x * 1000) for x in target_xyz_m]
    target_rpy_rad = [float(np.radians(a)) for a in target_rpy_deg]

    # xyz in meters, rx ry rz in radians
    target_pose = target_xyz_m + target_rpy_rad
    print("Target TCP position (m, rad):", target_pose)

    current_tcp_pose = robot.get_tcp_pose() 
    print("Current TCP position (m, rad):", current_tcp_pose)  
   
    current_joint_angles = robot.get_actual_joints_position()
    # print(f"Current joint angles (rad): {current_joint_angles}")
    print(f"Current joint angles (deg): {[float(np.degrees(a)) for a in current_joint_angles]}")
    
    try:
        target_joint_angles = robot.cal_ikine(target_pose, '', '', '') # in radians
        print(f"Target joint angles (deg): {[float(np.degrees(a)) for a in target_joint_angles]}")
        return target_joint_angles
    
    except Exception as e:
        print(f"Inverse kinematics failed: {e}")
        return None


def get_actual_tcp_pose_from_joint_angles(robot, joint_angles):
    """
    Calculate the actual TCP pose that will be achieved with the given joint angles.
    This uses forward kinematics to get the real TCP pose after inverse kinematics solving.
    
    Args:
        robot: DucoCobot instance
        joint_angles: Target joint angles in radians
        
    Returns:
        tuple: (position_xyz, rpy_angles, rotation_matrix) or None if failed
            - position_xyz: [x, y, z] in meters  
            - rpy_angles: [rx, ry, rz] in degrees (ZYX intrinsic rpy angles)
            - rotation_matrix: 3x3 rotation matrix
    """
    if joint_angles is None:
        return None
        
    try:
        # Use forward kinematics to calculate actual TCP pose from joint angles
        actual_tcp_pose = robot.cal_fkine(joint_angles,'','')  # Returns [x, y, z, rx, ry, rz]
        
        # Extract position and orientation
        actual_pos = actual_tcp_pose[:3]  # [x, y, z] in meters
        actual_rpy_rad = actual_tcp_pose[3:6]  # [rx, ry, rz] in radians
        actual_rpy_deg = [float(np.degrees(a)) for a in actual_rpy_rad]
        
        # Convert rx, ry, rz to rotation matrix for visualization
        rx, ry, rz = actual_rpy_rad
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)
        
        actual_rot_matrix = np.array([
            [cos_rz*cos_ry, cos_rz*sin_ry*sin_rx - sin_rz*cos_rx, cos_rz*sin_ry*cos_rx + sin_rz*sin_rx],
            [sin_rz*cos_ry, sin_rz*sin_ry*sin_rx + cos_rz*cos_rx, sin_rz*sin_ry*cos_rx - cos_rz*sin_rx],
            [-sin_ry, cos_ry*sin_rx, cos_ry*cos_rx]
        ])
        
        print(f"Actual TCP position from FK (m): {actual_pos}")
        print(f"Actual TCP orientation from FK (rad): {actual_rpy_rad}")
        
        return actual_pos, actual_rpy_deg, actual_rot_matrix
        
    except Exception as e:
        print(f"Forward kinematics failed: {e}")
        return None


def calculate_rotation_axis_angles(R1, R2):
    """
    Calculate the angle differences between corresponding axes of two rotation matrices.
    
    Args:
        R1: First rotation matrix (3x3) - Ideal frame
        R2: Second rotation matrix (3x3) - Actual frame
    
    Returns:
        tuple: (x_angle_diff, y_angle_diff, z_angle_diff) in degrees
    """
    # Extract axes from rotation matrices
    # Each column represents an axis: [X_axis, Y_axis, Z_axis]
    x1, y1, z1 = R1[:, 0], R1[:, 1], R1[:, 2]  # Ideal frame axes
    x2, y2, z2 = R2[:, 0], R2[:, 1], R2[:, 2]  # Actual frame axes
    
    # Calculate angle between corresponding axes using dot product
    # angle = arccos(dot(v1, v2) / (|v1| * |v2|))
    def angle_between_vectors(v1, v2):
        # Ensure vectors are normalized
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate cosine of angle, clamp to [-1, 1] to avoid numerical errors
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        
        # Return angle in degrees
        return np.degrees(np.arccos(cos_angle))
    
    x_angle_diff = angle_between_vectors(x1, x2)
    y_angle_diff = angle_between_vectors(y1, y2)
    z_angle_diff = angle_between_vectors(z1, z2)
    
    return x_angle_diff, y_angle_diff, z_angle_diff


def draw_coordinate_frame(ax, origin, rotation_matrix, label, size=0.1, alpha=0.7):
    """
    Draw a coordinate frame (X-Y-Z axes) at specified origin with given rotation.
    
    Coordinate axis color convention:
    - RED arrow: X-axis (forward/backward direction)
    - GREEN arrow: Y-axis (left/right direction)  
    - BLUE arrow: Z-axis (up/down direction)
    
    Parameters:
    - ax: matplotlib 3D axes
    - origin: [x, y, z] position of frame origin
    - rotation_matrix: 3x3 rotation matrix
    - label: string label for the frame
    - size: length of axes arrows
    - alpha: transparency of arrows
    """
    # Define unit vectors for X, Y, Z axes
    x_axis = np.array([1, 0, 0]) * size
    y_axis = np.array([0, 1, 0]) * size
    z_axis = np.array([0, 0, 1]) * size
    
    # Rotate axes according to rotation matrix
    x_rotated = rotation_matrix @ x_axis
    y_rotated = rotation_matrix @ y_axis
    z_rotated = rotation_matrix @ z_axis
    
    # Draw axes as arrows with thicker lines and better arrow heads
    ax.quiver(origin[0], origin[1], origin[2], 
              x_rotated[0], x_rotated[1], x_rotated[2], 
              color='red', alpha=alpha, arrow_length_ratio=0.15, linewidth=3,
              normalize=False)
    ax.quiver(origin[0], origin[1], origin[2], 
              y_rotated[0], y_rotated[1], y_rotated[2], 
              color='green', alpha=alpha, arrow_length_ratio=0.15, linewidth=3,
              normalize=False)
    ax.quiver(origin[0], origin[1], origin[2], 
              z_rotated[0], z_rotated[1], z_rotated[2], 
              color='blue', alpha=alpha, arrow_length_ratio=0.15, linewidth=3,
              normalize=False)
    
    # Add axis labels at the end of each arrow
    ax.text(origin[0] + x_rotated[0] * 1.2, origin[1] + x_rotated[1] * 1.2, origin[2] + x_rotated[2] * 1.2, 
            'X', color='red', fontsize=12, fontweight='bold')
    ax.text(origin[0] + y_rotated[0] * 1.2, origin[1] + y_rotated[1] * 1.2, origin[2] + y_rotated[2] * 1.2, 
            'Y', color='green', fontsize=12, fontweight='bold')
    ax.text(origin[0] + z_rotated[0] * 1.2, origin[1] + z_rotated[1] * 1.2, origin[2] + z_rotated[2] * 1.2, 
            'Z', color='blue', fontsize=12, fontweight='bold')
    
    # Add frame label with offset
    label_offset = np.array([0, 0, size * 1.5])
    ax.text(origin[0] + label_offset[0], origin[1] + label_offset[1], origin[2] + label_offset[2], 
            label, fontsize=10, fontweight='bold', ha='center')


def visualize_coordinate_systems(robot=None, target_joint_angles=None):
    """
    Visualize the coordinate systems and key points involved in the robot positioning:
    - Base coordinate system (robot base)
    - Cabinet coordinate system 
    - Ideal task frame (from coordinate transformation)
    - Actual TCP target frame (from forward kinematics of solved joint angles)
    - Current TCP position (if robot is connected)
    
    Parameters:
    - robot: DucoCobot instance (optional, if provided will show current TCP)
    - target_joint_angles: Target joint angles from inverse kinematics (optional)
    """
    # Read coordinate transformation data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    coord_file = os.path.join(project_root, 'test_data', 'output', 'coordinate_system_results.json')
    pose_file = os.path.join(project_root, 'test_data', 'positioning_data', 'task_start_pose.json')
    
    try:
        with open(coord_file, 'r') as f:
            coord_data = json.load(f)
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Required data file not found: {e}")
        return
    
    # Get transformation matrix from cabinet to base
    cabinet2base = np.array(coord_data['estimated_cabinet2base_matrix'])
    
    # Get task pose in cabinet coordinate
    task_pos_cabinet = np.array(pose_data['position'])
    task_rot_cabinet = np.array(pose_data['rotation_matrix'])
    
    # Calculate task pose in base coordinate - get both position and rotation matrix directly
    target_xyz_base, target_rpy_deg = get_start_point_in_base()
    task_pos_base = np.array(target_xyz_base)
    
    # Get the correct task rotation matrix by directly transforming from cabinet to base
    # Create 4x4 transformation matrix for task pose in cabinet
    task_matrix_cabinet = np.eye(4)
    task_matrix_cabinet[:3, :3] = task_rot_cabinet
    task_matrix_cabinet[:3, 3] = task_pos_cabinet
    
    # Transform to base coordinate: T_base = T_cabinet2base * T_task_cabinet
    task_matrix_base = cabinet2base @ task_matrix_cabinet
    
    # Extract the correct rotation matrix from the transformed matrix
    task_rot_base = task_matrix_base[:3, :3]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate workspace bounds
    all_points = [
        np.array([0, 0, 0]),  # base origin
        cabinet2base[:3, 3],  # cabinet origin
        task_pos_base,        # task in base
    ]
    
    # Add current TCP if available
    current_tcp_pos = None
    current_tcp_full = None
    if robot is not None:
        try:
            current_tcp_full = robot.get_tcp_pose()  # [x, y, z, rx, ry, rz]
            current_tcp_pos = np.array(current_tcp_full[:3])
            all_points.append(current_tcp_pos)
        except Exception as e:
            print(f"Failed to get TCP pose: {e}")
    
    # Calculate bounds with some padding
    all_coords = np.array(all_points)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
    
    padding = 0.1
    x_range = max(x_max - x_min + 2*padding, 0.2)
    y_range = max(y_max - y_min + 2*padding, 0.2)
    z_range = max(z_max - z_min + 2*padding, 0.2)
    
    # Draw base coordinate system (origin at robot base)
    base_origin = np.array([0.0, 0.0, 0.0])
    base_rotation = np.eye(3)  # Identity matrix for base frame
    draw_coordinate_frame(ax, base_origin, base_rotation, 'Base Frame', size=0.08, alpha=0.8)
    
    # Draw cabinet coordinate system
    cabinet_origin = cabinet2base[:3, 3]
    cabinet_rotation = cabinet2base[:3, :3]
    draw_coordinate_frame(ax, cabinet_origin, cabinet_rotation, 'Cabinet Frame', size=0.08, alpha=0.8)
    
    # Draw ideal task frame (from coordinate transformation)
    draw_coordinate_frame(ax, task_pos_base, task_rot_base, 
                         'Ideal Task Frame', size=0.08, alpha=0.8)
    
    # Draw actual TCP target frame (from forward kinematics) if joint angles are available
    actual_tcp_info = None
    if robot is not None and target_joint_angles is not None:
        actual_tcp_info = get_actual_tcp_pose_from_joint_angles(robot, target_joint_angles)
        if actual_tcp_info is not None:
            actual_tcp_pos, actual_tcp_rpy, actual_tcp_rot = actual_tcp_info
            actual_tcp_pos_array = np.array(actual_tcp_pos)
            all_points.append(actual_tcp_pos_array)
            
            # Draw actual TCP target frame with different styling and slight offset for visibility
            # Add small offset to make it visible when overlapping with ideal frame
            offset = np.array([0.01, 0.01, 0.01])  # 1cm offset for visibility
            draw_coordinate_frame(ax, actual_tcp_pos_array + offset, actual_tcp_rot, 
                                 'Actual Target Pose', size=0.06, alpha=0.9)
            
            print(f"Actual Target Pose frame displayed with {offset} offset for visibility")
            
            # Draw line connecting ideal and actual positions to show difference
            ax.plot([task_pos_base[0], actual_tcp_pos[0]], 
                   [task_pos_base[1], actual_tcp_pos[1]], 
                   [task_pos_base[2], actual_tcp_pos[2]], 
                   'r--', alpha=0.7, linewidth=2, label='Ideal↔Actual Deviation')
    
    # Draw current TCP frame if available
    if current_tcp_pos is not None and current_tcp_full is not None:
        # Get current TCP orientation and draw frame
        current_tcp_rxryrz = current_tcp_full[3:6]  # rx, ry, rz in radians
        
        # Convert rx, ry, rz to rotation matrix for current TCP
        rx, ry, rz = current_tcp_rxryrz
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)
        
        current_tcp_rot = np.array([
            [cos_rz*cos_ry, cos_rz*sin_ry*sin_rx - sin_rz*cos_rx, cos_rz*sin_ry*cos_rx + sin_rz*sin_rx],
            [sin_rz*cos_ry, sin_rz*sin_ry*sin_rx + cos_rz*cos_rx, sin_rz*sin_ry*cos_rx - cos_rz*sin_rx],
            [-sin_ry, cos_ry*sin_rx, cos_ry*cos_rx]
        ])
        
        # Draw current TCP frame with same style as other frames
        draw_coordinate_frame(ax, current_tcp_pos, current_tcp_rot, 
                             'Current TCP', size=0.08, alpha=0.8)
    
    # Connect cabinet origin to base origin with line
    ax.plot([0, cabinet_origin[0]], [0, cabinet_origin[1]], [0, cabinet_origin[2]], 
            'k--', alpha=0.5, linewidth=1)
    
    # Create custom legend for coordinate axes colors
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='X-axis')
    green_patch = mpatches.Patch(color='green', label='Y-axis')
    blue_patch = mpatches.Patch(color='blue', label='Z-axis')
    
    # Create legend with coordinate axis colors
    ax.legend(handles=[red_patch, green_patch, blue_patch], 
             title='Coordinate Axes', loc='upper right')
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('Robot Coordinate Systems and Task Points', fontsize=14, fontweight='bold')
    
    # Set bounds based on actual data
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2
    
    max_range = max(x_range, y_range, z_range) / 2
    ax.set_xlim([center_x - max_range, center_x + max_range])
    ax.set_ylim([center_y - max_range, center_y + max_range])
    ax.set_zlim([max(0, center_z - max_range), center_z + max_range])
    
    # Add grid with better styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make pane edges more visible
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Set better viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save the plot to file instead of showing it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, 'test_data', 'output', 'frame_results.jpg')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Frame results saved to: {output_path}")
    plt.close()  # Close the figure to free memory
    
    # Print transformation information
    print("\n=== Coordinate System Information ===")
    print("Coordinate Axis Color Convention:")
    print("  RED arrows (X)   = X-axis (forward/backward direction)")
    print("  GREEN arrows (Y) = Y-axis (left/right direction)")
    print("  BLUE arrows (Z)  = Z-axis (up/down direction)")
    print("")
    print("Coordinate Systems Analysis:")
    print(f"Cabinet origin in base frame: {cabinet_origin}")
    print(f"Task point in base frame: {task_pos_base}")
    print(f"Task orientation (rx,ry,rz deg): {target_rpy_deg}")

    
    # Calculate relative orientations
    cabinet_rot = cabinet2base[:3, :3]
    
    # Calculate Cabinet Frame orientation relative to Base Frame
    cabinet_rpy = []
    R = cabinet_rot
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0
    cabinet_rpy = [float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))]
    
    print(f"Cabinet Frame orientation relative to Base (rx,ry,rz deg): {[round(x, 2) for x in cabinet_rpy]}")
    print(f"Ideal Task Frame orientation relative to Base (rx,ry,rz deg): {[round(x, 2) for x in target_rpy_deg]}")
    
    # Show comparison between ideal and actual TCP target if available
    if actual_tcp_info is not None:
        actual_tcp_pos, actual_tcp_rpy, actual_tcp_rot = actual_tcp_info
        
        print(f"\n=== Ideal vs Actual Target Pose Comparison ===")
        
        # 1. Origin Position Comparison
        print(f"1. Origin Position Comparison:")
        print(f"   Ideal frame origin (m):    {[round(float(x), 6) for x in task_pos_base]}")
        print(f"   Actual frame origin (m):   {[round(float(x), 6) for x in actual_tcp_pos]}")
        
        # Calculate position deviation
        pos_deviation = np.array(actual_tcp_pos) - np.array(task_pos_base)
        pos_deviation_norm = np.linalg.norm(pos_deviation)
        print(f"   Position deviation (m):     {[round(float(x), 6) for x in pos_deviation]}")
        print(f"   Position deviation norm:    {pos_deviation_norm:.6f} m")
        
        if pos_deviation_norm < 0.001:  # Less than 1mm
            print("   ✓ Position accuracy: EXCELLENT (< 1mm)")
        elif pos_deviation_norm < 0.005:  # Less than 5mm
            print("   ✓ Position accuracy: GOOD (< 5mm)")
        else:
            print("   ⚠ Position accuracy: CHECK REQUIRED (> 5mm)")
        
        # 2. Axes Orientation Comparison
        print(f"\n2. Coordinate Axes Angular Differences:")
        
        # Get rotation matrices
        ideal_rot_matrix = task_rot_base
        actual_rot_matrix = actual_tcp_rot
        
        # Calculate axis angle differences
        x_angle_diff, y_angle_diff, z_angle_diff = calculate_rotation_axis_angles(ideal_rot_matrix, actual_rot_matrix)
        
        print(f"   X-axis angular difference:  {x_angle_diff:.4f}°")
        print(f"   Y-axis angular difference:  {y_angle_diff:.4f}°") 
        print(f"   Z-axis angular difference:  {z_angle_diff:.4f}°")
        
        # Overall orientation accuracy assessment
        max_axis_deviation = max(x_angle_diff, y_angle_diff, z_angle_diff)
        if max_axis_deviation < 0.1:  # Less than 0.1 degrees
            print("   ✓ Orientation accuracy: EXCELLENT (< 0.1°)")
        elif max_axis_deviation < 1.0:  # Less than 1 degree
            print("   ✓ Orientation accuracy: GOOD (< 1.0°)")
        else:
            print("   ⚠ Orientation accuracy: CHECK REQUIRED (> 1.0°)")
    else:
        print(f"\n(Actual Target Pose frame not available - no joint angles calculated)")


def main():
    # Create the DucoCobot instance and open connection  
    robot = DucoCobot(ip, port)  
    res = robot.open()  
    print("Open connection:", res)  
        
    # # Power on and enable the robot  
    # res = robot.power_on(True)  
    # print("Power on:", res)  
    # res = robot.enable(True)  
    # print("Enable:", res)  
        
    # Set up an Op instance with no triggering events (default)  
    op = Op()  
    op.time_or_dist_1 = 0  
    op.trig_io_1 = 1  
    op.trig_value_1 = False  
    op.trig_time_1 = 0.0  
    op.trig_dist_1 = 0.0  
    op.trig_event_1 = ""  
    op.time_or_dist_2 = 0  
    op.trig_io_2 = 1  
    op.trig_value_2 = False  
    op.trig_time_2 = 0.0  
    op.trig_dist_2 = 0.0  
    op.trig_event_2 = ""  
        
    # ====================================================================           
    target_joints_positions_rad = None
    try:
        # Calculate target joint angles using inverse kinematics
        target_joints_positions_rad = calculate_target_joint_angles(robot)
        
        # if target_joints_positions_rad is not None:
        #     print("Moving to Target joint angles!")
        #     res = robot.movej2(target_joints_positions_rad, 0.5, 1.0, 0.0, True, op)
        #     time.sleep(0.5)
        # else:
        #     print("Failed to calculate target joint angles")
        
    except Exception as e:
        print(f"Error in calculation: {e}")
        
    # ====================================================================
    # Visualize coordinate systems and points
    print("\n=== Visualizing Coordinate Systems ===")
    visualize_coordinate_systems(robot, target_joints_positions_rad)
        
    # ====================================================================
    # # Clean up: disable, power off, and close connection  
    # res = robot.disable(True)  
    # print("\nDisable result:", res)  
    # res = robot.power_off(True)  
    # print("Power off result:", res)  
    res = robot.close()  
    print("Close connection result:", res)  


if __name__ == "__main__":
    main()
