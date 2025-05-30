import numpy as np

q_init = [-6, -6, -12]
q_goal = [0, 8, -12]

def get_simulation_config(q_init, q_goal):
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
