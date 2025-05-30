import holoocean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time

from config import get_simulation_config, q_init, q_goal
from planning import setup_planners, get_track_list
from visualization import (
    setup_plot, update_airplane_icon, update_plot_data
)
from control import (
    get_states_6dof, rotate_6dof_forces, rotate_velocities_g_b,
    compute_thruster_commands, distance
)

# Initialize simulation configuration
cfg = get_simulation_config(q_init, q_goal)
start_time = time.time()

def main():
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
            # Get current and next waypoint for local path
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
            update_airplane_icon(ax, plot_handles['airplane_patch'], pose)
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