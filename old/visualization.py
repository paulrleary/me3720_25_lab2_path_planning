import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def setup_plot(track_list):
    fig, ax = plt.subplots()
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Real-time Plot")
    
    # Plot track and BOP obstacle
    track_x_data = [wp[0] for wp in track_list]
    track_y_data = [wp[1] for wp in track_list]
    track_line, = ax.plot(track_x_data, track_y_data, 'r--o', label='Track')
    
    bop_x_data = [-4, -4, 7, 7, -4]
    bop_y_data = [-4, 7, 7, -4, -4]
    bop_line, = ax.plot(bop_x_data, bop_y_data, 'k-', label='BOP')
    
    # Initialize airplane icon
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

def update_airplane_icon(ax, airplane_patch, pose):
    airplane_shape = np.array([
        [0.5, 0.0], [-0.5, 0.2], [-0.3, 0.0], [-0.5, -0.2], [0.5, 0.0]
    ])
    c, s = np.cos(np.deg2rad(pose[5])), np.sin(np.deg2rad(pose[5]))
    R = np.array([[c, -s], [s, c]])
    airplane_xy = (R @ airplane_shape.T).T + [pose[0], pose[1]]
    airplane_patch.set_xy(airplane_xy)

def update_plot_data(plot_handles, pos_x_data, pos_y_data):
    plot_handles['position_line'].set_xdata(pos_x_data)
    plot_handles['position_line'].set_ydata(pos_y_data)
    plot_handles['position_line'].axes.relim()
    plot_handles['position_line'].axes.autoscale_view()
    plot_handles['position_line'].figure.canvas.draw()
    plot_handles['position_line'].figure.canvas.flush_events()