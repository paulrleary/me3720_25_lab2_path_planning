import holoocean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time
from scipy.spatial.transform import Rotation

# --- CONFIGURATION SECTION ---

def get_simulation_config(q_init, q_goal):
    """
    Returns the simulation configuration dictionary for HoloOcean.
    """
    heading_init = np.degrees(np.arctan2(q_goal[1] - q_init[1], q_goal[0] - q_init[0]))
    cfg = {
        "name": "test_rgb_camera",
        "world": "BlowoutPreventerSampleLevel",
        "package_name": "USS_Environ",
        "main_agent": "auv0",
        "ticks_per_sec": 10,
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "HoveringAUV",
                "sensors": [{"sensor_type": "DynamicsSensor"}],
                "control_scheme": 2,
                "location": q_init,
                "rotation": [0, 0, heading_init]
            }
        ]
    }
    return cfg

# --- PLANNING SECTION ---

def setup_planners(q_init, q_goal):
    """
    Initializes and runs path planners. Returns the planned paths.
    """
    obstacles = [[(-4, -4), (-4, 7), (7, 7), (7, -4)]]
    start = (q_init[0], q_init[1])
    goal = (q_goal[0], q_goal[1])

    # These classes must be defined/imported elsewhere
    astar_planner = AStarPlanner(obstacles, start, goal, grid_res=0.2, max_iter=2000)
    rrt_star_planner = RRTStarPlanner(obstacles, start, goal, step_size=1.0, neighbor_radius=2.0, max_iter=2000)
    astar_tree, astar_path = astar_planner.plan()
    rrt_star_nodes, rrt_star_path = rrt_star_planner.plan()
    return astar_path, rrt_star_path

def get_track_list(rrt_star_path):
    """
    Converts a 2D path to a list of 3D waypoints (with constant depth).
    """
    return [np.array([*wp, 0]) for wp in rrt_star_path]

# --- VISUALIZATION SECTION ---

def setup_plot(track_list):
    """
    Sets up the matplotlib plot and returns figure, axis, and plot handles.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Real-time Plot")
    
    # Plot the planned track and BOP obstacle
    track_x_data = [wp[0] for wp in track_list]
    track_y_data = [wp[1] for wp in track_list]
    track_line, = ax.plot(track_x_data, track_y_data, 'r--o', label='Track')
    
    bop_x_data = [-4, -4, 7, 7, -4]
    bop_y_data = [-4, 7, 7, -4, -4]
    bop_line, = ax.plot(bop_x_data, bop_y_data, 'k-', label='BOP')
    
    # Initialize vehicle icon
    airplane_shape = np.array([
        [0.5, 0.0], [-0.5, 0.2], [-0.3, 0.0], [-0.5, -0.2], [0.5, 0.0]
    ])
    start_pose = [track_list[0][0], track_list[0][1], 0]
    c, s = np.cos(np.deg2rad(start_pose[2])), np.sin(np.deg2rad(start_pose[2]))
    R = np.array([[c, -s], [s, c]])
    airplane_xy = (R @ airplane_shape.T).T + [start_pose[0], start_pose[1]]
    airplane_patch = Polygon(airplane_xy, closed=True, color='blue', alpha=0.7)
    ax.add_patch(airplane_patch)
    
    position_line, = ax.plot([], [], label='AUV Path')
    plt.ion()
    return fig, ax, {
        'track_line': track_line,
        'bop_line': bop_line,
        'airplane_patch': airplane_patch,
        'position_line': position_line
    }

def update_airplane_icon(airplane_patch, pose):
    """
    Updates the orientation and position of the vehicle icon.
    """
    airplane_shape = np.array([
        [0.5, 0.0], [-0.5, 0.2], [-0.3, 0.0], [-0.5, -0.2], [0.5, 0.0]
    ])
    c, s = np.cos(np.deg2rad(pose[5])), np.sin(np.deg2rad(pose[5]))
    R = np.array([[c, -s], [s, c]])
    airplane_xy = (R @ airplane_shape.T).T + [pose[0], pose[1]]
    airplane_patch.set_xy(airplane_xy)

def update_plot_data(plot_handles, pos_x_data, pos_y_data):
    """
    Updates the path trace on the plot.
    """
    plot_handles['position_line'].set_xdata(pos_x_data)
    plot_handles['position_line'].set_ydata(pos_y_data)
    ax = plot_handles['position_line'].axes
    ax.relim()
    ax.autoscale_view()
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

# --- CONTROL & UTILITY SECTION ---

def get_states_6dof(dynamics_sensor_output_rpy, start_time):
    """
    Extracts pose, velocity, acceleration, and time from simulation output.
    """
    x = dynamics_sensor_output_rpy
    a, v, p, alpha, omega, theta = x[:3], x[3:6], x[6:9], x[9:12], x[12:15], x[15:18]
    pos = np.concatenate((p, theta))
    vel = np.concatenate((v, omega))
    acc = np.concatenate((a, alpha))
    t = time.time() - start_time
    return dict(pose=pos, velocity=vel, acceleration=acc, time=t)

def rotate_6dof_forces(forces, eulers):
    """
    Rotates force/torque vector from body to global frame.
    """
    R = Rotation.from_euler('xyz', eulers, degrees=True).as_matrix()
    rotated_forces = np.zeros(6)
    rotated_forces[0:3] = R @ forces[0:3]
    rotated_forces[3:6] = R @ forces[3:6]
    return rotated_forces

def rotate_velocities_g_b(velocities, eulers):
    """
    Rotates velocity from global to body frame.
    """
    R = Rotation.from_euler('xyz', eulers, degrees=True).as_matrix()
    R_inv = R.T
    return R_inv @ velocities

def compute_thruster_commands(pose, vel, current_path, t):
    """
    Computes thruster commands and global acceleration based on current state and path following.
    """
    # Cross-track error and path following
    normal_error, binormal_error, frenet_serret_frame = cte(pose[:3], current_path[0], current_path[1])
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
    """
    Returns Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- MAIN SIMULATION LOOP ---

def main():
    # Initial and goal positions
    q_init = [-6, -6, -12]
    q_goal = [0, 8, -12]
    cfg = get_simulation_config(q_init, q_goal)
    start_time = time.time()

    # Path planning
    astar_path, rrt_star_path = setup_planners(q_init, q_goal)
    track_list = get_track_list(rrt_star_path)
    
    # Visualization setup
    fig, ax, plot_handles = setup_plot(track_list)
    pos_x_data, pos_y_data = [], []

    # Simulation loop
    with holoocean.make(scenario_cfg=cfg) as env:
        goal_idx = 1
        acceleration_global = np.zeros(6)
        while True:
            # Get current and next waypoint for local path following
            current_path = [track_list[goal_idx - 1], track_list[goal_idx]]
            
            # Step simulation
            simul_state = env.step(acceleration_global)
            auv_state = get_states_6dof(simul_state["DynamicsSensor"], start_time)
            pose, vel, t = auv_state["pose"], auv_state["velocity"], auv_state["time"]
            
            # Compute control and thruster commands
            thruster_command, acceleration_global = compute_thruster_commands(
                pose, vel, current_path, t
            )
            
            # Visualization updates
            pos_x_data.append(pose[0])
            pos_y_data.append(pose[1])
            update_airplane_icon(plot_handles['airplane_patch'], pose)
            update_plot_data(plot_handles, pos_x_data, pos_y_data)
            plt.pause(0.01)
            
            # Check if waypoint reached
            D = distance(pose[:2], current_path[1][:2])
            if D < 1:
                goal_idx += 1
                if goal_idx >= len(track_list):
                    print('Finished track!')
                    break
                else:
                    print('Goal reached! Next Waypoint!')

if __name__ == "__main__":
    main()
