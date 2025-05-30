import numpy as np
import time

def vert_force_to_thrusters(force):
    ind_force = force/4
    
    command = np.zeros(8)
    command[0:4] = ind_force

    return command

# def yaw_force_to_thrusters(force):
#     ind_force = force/4
    
#     command = np.zeros(8)
#     command[4] = -1*ind_force
#     command[5] = ind_force
#     command[6] = ind_force
#     command[7] = -1*ind_force

def yaw_force_to_thrusters(force):
    ind_force = force/4
    
    command = np.zeros(8)
    command[4] = ind_force
    command[5] = -1*ind_force
    command[6] = -1*ind_force
    command[7] = -ind_force

    return command

def speed_force_to_thrusters(force):
    ind_force = force/4
    
    command = np.zeros(8)
    command[4:8] = ind_force
  
    return command

def control_angle_delta_degrees(current_angle, set_angle):
    """
    Calculate the difference between two angles in degrees.
    
    Parameters:
    current_angle (float): First angle in degrees.
    set_angle (float): Second angle in degrees.
    
    Returns:
    delta_angle (float): The difference between the two angles in degrees.
    """
    delta_angle = (current_angle - set_angle + 180) % 360 - 180
    return delta_angle

from scipy.spatial.transform import Rotation
def rotate_6dof_forces(forces, eulers):
    R=Rotation.from_euler('xyz', eulers, degrees=True).as_matrix()
    rotated_forces = np.zeros(6)
    rotated_forces[0:3] = R@forces[0:3]
    rotated_forces[3:6] = R@forces[3:6]

    return rotated_forces

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def frenet_seret_frame_to_heading_command(lookahead_distance, normal_error, frenet_serret_frame):
    T=frenet_serret_frame[0]
    N=frenet_serret_frame[1]
    new_frame=lookahead_distance*T-normal_error*N
    goal_heading = np.arctan2(new_frame[1],new_frame[0])*180/np.pi
    return goal_heading

start_time = time.time()
def get_states_6dof(dynamics_sensor_output_rpy):
    x = dynamics_sensor_output_rpy # dynamics_sensor_output_rpy is a 18-element array containing the state of the AUV
    
    a = x[:3] # linear acceleration in global frame
    v = x[3:6] # linear velocity in global frame
    p = x[6:9] # position in global frame
    alpha = x[9:12] # angular acceleration in global frame
    omega = x[12:15] # angular velocity in global frame
    theta = x[15:18] # orientation in global frame (roll, pitch, yaw)

    pos = np.concatenate((p,theta))
    vel = np.concatenate((v,omega))
    acc = np.concatenate((a,alpha))
    t = time.time() - start_time

    state_6dof = dict(pose=pos, velocity=vel,acceleration=acc, time=t)

    return state_6dof


def rotate_velocities_global_body(velocities, eulers):
    R=Rotation.from_euler('xyz', eulers, degrees=True).as_matrix()
    R_inv = R.T
    rotated_velocities = np.zeros(3)
    rotated_velocities = R_inv@velocities
 
    return rotated_velocities

def get_holocean_config_BOP(q_init, heading_init):
    """
    Create a configuration dictionary for the HoloOcean simulation.
    """
    return {
        "name": "test_rgb_camera",
        "world": "BlowoutPreventerSampleLevel",
        "package_name": "USS_Environ",
        "main_agent": "auv0",
        "ticks_per_sec": 10,
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "HoveringAUV",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                    },
                ],
                "control_scheme": 2,
                "location": q_init,
                "rotation": [0, 0, heading_init],
            }
        ]
    }