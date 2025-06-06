import holoocean
import numpy as np
# from pynput import keyboard
# import cv2
# cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

from scipy.spatial.transform import Rotation
np.set_printoptions(suppress=True) # Suppress scientific notation in printing
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import time

# import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'clutter'))


from HoveringAUV_Physical_Model import global_hydro_forces, thrusters_to_body_forces, forces_to_accelerations
# from Frenet_Serret_Error import frenet_seret_cross_track_error as frenet_seret_cte
# from my_PID_controllers import depth_pid_controller, heading_pid_controller, speed_pid_controller, lookahead_distance
from helper_functions import (
    vert_force_to_thrusters, yaw_force_to_thrusters, speed_force_to_thrusters, 
    control_angle_delta_degrees, rotate_6dof_forces, distance,closest_waypoint_index,
    frenet_seret_frame_to_heading_command as frenet_seret_heading, heading_to_point,
    get_states_6dof, rotate_velocities_global_body as rotate_velocities_g_b,
    get_holocean_config_BOP_april_tags as get_config,
)

from error_monitors import DistanceErrorMonitor as distance_monitor

from plot_helpers import get_airplane_xy, get_airplane_pose

from path_planners import RRTPlanner, RRTStarPlanner, AStarPlanner

# q_init = [-6, -6, -12]
# q_init = [-6, 4, -12]
q_init = [-5, 4, -12]
q_goal = [0, 8, -12]

heading_init = np.atan2(q_goal[1] - q_init[1], q_goal[0] - q_init[0]) * 180 / np.pi


    
# Create the configuration dictionary for HoloOcean
cfg = get_config(q_init, heading_init)

# Define the BOP (Blowout Preventer) as a rectangular obstacle.  The Inflated version extends the obstacle by 1 meter in all directions, for easier collision avoidance.
# BOPstacle = [(-3, -3), (-3,6), (6,6), (6, -3)]
BOPstacle_inflated = [(-4, -4), (-4,7), (7,7), (7, -4)]
BOP_corners = BOPstacle_inflated

## Initialize the plot
plt.ion()
pos_x_data = []
pos_y_data = []

track_x_data = []
track_y_data = []

bop_x_data = []
bop_y_data = []

## Add the BOP to the plot
for wp in BOP_corners:
    bop_x_data.append(wp[0])
    bop_y_data.append(wp[1])
bop_x_data.append(BOP_corners[0][0])
bop_y_data.append(BOP_corners[0][1])

fig, ax = plt.subplots()

track_line, = ax.plot([], [], 'r--o', label='Track')
path_line, = ax.plot([], [], 'g-', label='Current')
bop_line, = ax.plot(bop_x_data, bop_y_data, 'k-', label='BOP')
bop_line.set_xdata(bop_x_data)
bop_line.set_ydata(bop_y_data)

position_line, = ax.plot(pos_x_data, pos_y_data)

# Initialize airplane icon at the origin
start_pose = cfg["agents"][0]["location"]+cfg["agents"][0]["rotation"]
airplane_patch = Polygon(get_airplane_xy(get_airplane_pose(start_pose)), closed=True, color='blue', alpha=0.7)
ax.add_patch(airplane_patch)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Real-time Path Following")


## Initialize the planner configuration variables
start = (q_init[0], q_init[1])
goal = (q_goal[0], q_goal[1])

step_size = 1.0
neighbor_radius = 2.0
max_iter = 2000
grid_res = 0.2

obstacles = [BOP_corners]
## Below is a clever trick, where we want to hold a random sampler constant, for reproducibility.
# Uncomment the lines below to set a random seed for reproducibility
import random
random_seed = 43  # Or any integer you want
random.seed(random_seed)
np.random.seed(random_seed)
## Create the planner objects and plan the paths
astar_planner = AStarPlanner(obstacles, start, goal,
                                grid_res=grid_res, max_iter=max_iter)
rrt_planner = RRTPlanner(obstacles, start, goal,
                            step_size=step_size, neighbor_radius=neighbor_radius, max_iter=max_iter)
rrt_star_planner = RRTStarPlanner(obstacles, start, goal,
                                    step_size=step_size,
                                    neighbor_radius=neighbor_radius,
                                    max_iter=max_iter)
astar_tree, astar_path = astar_planner.plan()
rrt_nodes, rrt_path = rrt_planner.plan()
rrt_star_nodes, rrt_star_path = rrt_star_planner.plan()

selected_path = rrt_star_path # Pick which path to use for the AUV to follow. Here we use the RRT* path, but you can also use the A* or RRT paths

## Define the track list for the AUV to follow
## Because the Planner outputs are in the form of (x, y) coordinates, we need to append a depth to them.
## As we will only actually follow the path in 2D, handling the depth separately, the path_depth value is not actually important
path_depth = -10
track_list = selected_path
track_list = [
    np.array([*wp, path_depth])
    for wp in track_list
]

## Add the track waypoints to the plot
for wp in track_list:
    track_x_data.append(wp[0])
    track_y_data.append(wp[1])

track_line.set_xdata(track_x_data)
track_line.set_ydata(track_y_data)


## START THE SIMULATION
with holoocean.make(scenario_cfg=cfg)as env:
    depth_command = -10
    
    goal_idx = 1
    
    ## Initialize the Thruster Commands and Forces
    hydro_forces_global = np.zeros(6)

    acceleration_global = np.zeros(6)

    thruster_command = np.zeros(8)
  
    from controllers import PIDController, frenet_seret_cross_track_error as frenet_seret_cte

    ### STUDENT UPDATE NUMBER 1: Update the PID controllers as you see fit
    
    depth_pid_controller = PIDController(Kp=4, Ki=0.1, Kd=3, setpoint=0)
    heading_pid_controller = PIDController(Kp=.5, Ki=0, Kd=0, setpoint=0)
    speed_pid_controller = PIDController(Kp=1, Ki=0, Kd=0, setpoint=0)

    lookahead_distance = 0.5

    watch_circle_radius = 0.5  # meters
    
    distance_monitor = distance_monitor()
    
    
    # path_following_state = 'aligning'
    path_following_state = 'following'
    
    
    
    while True:                
        current_path = [track_list[goal_idx-1],
                         track_list[goal_idx]]
        
        path_length = distance(current_path[0], current_path[1])
        distance_monitor.threshold = path_length+2*watch_circle_radius
            
        # Step simulation
        simul_state = env.step(acceleration_global)
      
        ## Get hydro forces from HoloOcean
        sensor_state = simul_state["DynamicsSensor"]
        
        hydro_forces_global = global_hydro_forces(sensor_state)
        
        ## Get the AUV state from the simulation
        auv_state = get_states_6dof(simul_state["DynamicsSensor"])
        
        pose = auv_state["pose"]
        vel = auv_state["velocity"]
        t = auv_state["time"]
      
        ## STUDENT UPDATE MAIN: Experiment with and update the logic below.
        #  
        # Implement a better path follower based on whatever conditional logic you see fit.
        # Think about using distances, errors, or anything else to update your setpoints.
        # You can also use alternate PID controllers depending on the situation.
        # That is, you can have one form of controller for heading when following a long path 
        # with small normal errors, and another form of controller when trying to get on a new path.
        
        ## Get distance to the next waypoint
        D_goal = distance(pose[:2], current_path[1][:2]) # distance to the next waypoint, horizontal only
        
        
        ## Increment the goal index if the AUV is close enough to the next waypoint
        if D_goal < watch_circle_radius:
            goal_idx += 1
            distance_monitor.reset()
            if goal_idx >= len(track_list):
                print('Finished track!')
                break
            else:
                print('Goal reached! Next Waypoint!')
        
        
        ## Calculate the depth control value
        depth=pose[2]
        # Notice here, I am controlling based on setpoint rather than error
        depth_pid_controller.setpoint = depth_command
        depth_control = depth_pid_controller.update(depth)
        depth_error = depth_pid_controller.current_error
        
        #### Define Errors And Control Logic
        
        ## Calculate the heading control value   
        yaw=pose[5]
        normal_error, binormal_error, frenet_serret_frame = frenet_seret_cte(pose[:3], current_path[0], current_path[1])
        goal_heading = frenet_seret_heading(lookahead_distance, normal_error, frenet_serret_frame)
        heading_error = control_angle_delta_degrees(yaw, goal_heading)
        if abs(heading_error) > 10:
            print("HEADING ERROR TOO HIGH")
            speed_command = 0
        else:
            print("DRIVING")
            speed_command = 0.5
            
        # match path_following_state:
        #     case 'following':
        
        #         D_goal_error_check.update(D_goal)
                
        #         if D_goal_error_check.is_above_threshold():
        #             print("ABOVE DISTANCE THRESH")
        #             new_goal_idx = closest_waypoint_index(track_list, pose[:3])
                    
        #             goal_heading = heading_to_point(pose[:2], track_list[new_goal_idx][:2])
        #             heading_error = control_angle_delta_degrees(yaw, goal_heading)
                    
        #             if abs(heading_error) > 10:
        #                 print("HEADING ERROR TOO HIGH")
        #                 speed_command = 0
        #             else:
        #                 speed_command = 0.5
                        
        #             if new_goal_idx != goal_idx:
        #                 goal_idx = new_goal_idx
        #                 D_goal_error_check.reset()
                
        #             pass # Check for next best waypoint
                
        #         else:
        #             if D_goal_error_check.is_growing():
        #                 print("DISTANCE ERROR GROWING")
        #                 goal_heading =  heading_to_point(pose[:2], current_path[1][:2])
        #             else:
        #                 # goal_heading = frenet_seret_heading(lookahead_distance, normal_error, frenet_serret_frame)
        #                 goal_heading =  heading_to_point(pose[:2], current_path[1][:2])
        #             heading_error = control_angle_delta_degrees(yaw, goal_heading)
                    
        #             if abs(heading_error) > 10:
        #                 print("HEADING ERROR TOO HIGH")
        #                 speed_command = 0
        #             else:
        #                 speed_command = 0.5
        
        # Notice here, I am controlling based on error rather than setpoint
        heading_control = heading_pid_controller.update(heading_error)
        
                
        ## Calculate the speed control value
        body_velocity = rotate_velocities_g_b(vel[:3], pose[3:6])
        forward_speed = body_velocity[0]  # Forward speed in the body frame (velocity in body frame x direction)
        
        # speed_command = 0.5
        speed_pid_controller.setpoint = speed_command
        # Notice here, I am controlling based on setpoint rather than error
        speed_control = speed_pid_controller.update(forward_speed)
        # speed_control = 0.5 # This can be used to test the speed control without PID, i.e. a constant speed command
        
        print(heading_control, speed_control) 
        ## Calculate the thruster commands based on the control values
        vert_thruster_command = vert_force_to_thrusters(depth_control)   
        heading_thruster_command = yaw_force_to_thrusters(heading_control)
        speed_thruster_command = speed_force_to_thrusters(speed_control)

        thruster_command = vert_thruster_command+heading_thruster_command+speed_thruster_command

        ## Calculate the net forces and accelerations in the global frame
        theta = pose[3:6]
        pos = pose[0:3]
        thrust_forces_body = thrusters_to_body_forces(thruster_command)
        thrust_forces_global = rotate_6dof_forces(thrust_forces_body, theta)
        net_force_global = hydro_forces_global + thrust_forces_global

        acceleration_global = forces_to_accelerations(net_force_global)
        
        ## UPdate the plot with current position
        path_line.set_xdata([current_path[0][0], current_path[1][0]])
        path_line.set_ydata([current_path[0][1], current_path[1][1]])
        
        airplane_patch.set_xy(get_airplane_xy(get_airplane_pose(pose)))
        
        pos_x_data.append(pose[0])
        pos_y_data.append(pose[1])
        position_line.set_xdata(pos_x_data)
        position_line.set_ydata(pos_y_data)

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

