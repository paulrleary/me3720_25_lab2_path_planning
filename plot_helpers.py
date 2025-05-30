import numpy as np

def get_airplane_pose(start_pose):   
    return [start_pose[0], start_pose[1], start_pose[5]]  # [x, y, yaw]
# airplane_pose = [start_pose[0], start_pose[1], start_pose[5]]  # [x, y, yaw]

def get_airplane_xy(pose):    
    c, s = np.cos(np.deg2rad(pose[2])), np.sin(np.deg2rad(pose[2]))
    R = np.array([[c, -s], [s, c]])
    airplane_shape = np.array([
        [0.5, 0.0],   # Nose
        [-0.5, 0.2],  # Left wing
        [-0.3, 0.0],  # Tail left
        [-0.5, -0.2], # Right wing
        [0.5, 0.0]    # Back to nose
    ])
    airplane_xy = (R @ airplane_shape.T).T + [pose[0], pose[1]]
    return airplane_xy 