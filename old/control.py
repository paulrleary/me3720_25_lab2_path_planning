mport numpy as np
from scipy.spatial.transform import Rotation
import time

from HoveringAUV_Physical_Model import global_hydro_forces, thrusters_to_body_forces, forces_to_accelerations
from Frenet_Serret_Error import frenet_seret_cross_track_error as frenet_seret_cte
from helper_functions import vert_force_to_thrusters, yaw_force_to_thrusters, speed_force_to_thrusters, control_angle_delta_degrees
from PID_lib import depth_pid_controller, heading_pid_controller, speed_pid_controller

def get_states_6dof(dynamics_sensor_output_rpy, start_time):
    x = dynamics_sensor_output_rpy
    a, v, p, alpha, omega, theta = x[:3], x[3:6], x[6:9], x[9:12], x[12:15], x[15:18]
    pos = np.concatenate((p, theta))
    vel = np.concatenate((v, omega))
    acc = np.concatenate((a, alpha))
    t = time.time() - start_time
    return dict(pose=pos, velocity=vel, acceleration=acc, time=t)

def rotate_6dof_forces(forces, eulers):
    R = Rotation.from_euler('xyz', eulers, degrees=True).as_matrix()
    rotated_forces = np.zeros(6)
    rotated_forces[0:3] = R @ forces[0:3]
    rotated_forces[3:6] = R @ forces[3:6]
    return rotated_forces

def rotate_velocities_g_b(velocities, eulers):
    R = Rotation.from_euler('xyz', eulers, degrees=True).as_matrix()
    R_inv = R.T
    return R_inv @ velocities

def compute_thruster_commands(pose, vel, current_path, t):
    # Cross-track error and path following
    normal_error, binormal_error, frenet_serret_frame = frenet_seret_cte(pose[:3], current_path[0], current_path[1])
    lookahead_distance = 0.01
    T, N = frenet_serret_frame[0], frenet_serret_frame[1]
    new_frame = lookahead_distance * T - normal_error * N
    goal_heading = np.degrees(np.arctan2(new_frame[1], new_frame[0]))
    
    # Depth control
    depth_command = -10
    depth_pid_controller.setpoint = depth_command
    depth_control = depth_pid_controller.update(pose[2])
    
    # Heading control
    yaw = pose[5]
    heading_error = control_angle_delta_degrees(yaw, goal_heading)
    heading_control = heading_pid_controller.update(heading_error)
    
    # Speed control
    body_velocity = rotate_velocities_g_b(vel[:3], pose[3:6])
    forward_speed = body_velocity[0]
    speed_command = 0.5
    speed_pid_controller.setpoint = speed_command
    speed_control = speed_pid_controller.update(forward_speed)
    
    # Thruster mapping
    vert_thruster_command = vert_force_to_thrusters(depth_control)
    heading_thruster_command = yaw_force_to_thrusters(heading_control)
    speed_thruster_command = speed_force_to_thrusters(speed_control)
    thruster_command = vert_thruster_command + heading_thruster_command + speed_thruster_command
    
    # Forces and acceleration
    hydro_forces_global = global_hydro_forces(pose)
    thrust_forces_body = thrusters_to_body_forces(thruster_command)
    thrust_forces_global = rotate_6dof_forces(thrust_forces_body, pose[3:6])
    net_force_global = hydro_forces_global + thrust_forces_global
    acceleration_global = forces_to_accelerations(net_force_global)
    
    return thruster_command, acceleration_global

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))