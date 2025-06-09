import holoocean
import holoocean.weather
import numpy as np
from pynput import keyboard
import cv2
import apriltag  # <-- AprilTag detector

from helper_functions import (
    vert_force_to_thrusters, yaw_force_to_thrusters, speed_force_to_thrusters, 
    control_angle_delta_degrees, rotate_6dof_forces, distance,
    frenet_seret_frame_to_heading_command as frenet_seret_heading,
    get_states_6dof, rotate_velocities_global_body as rotate_velocities_g_b,
    get_holocean_config_BOP as get_config,
)

np.set_printoptions(suppress=True) # Suppress scientific notation in printing

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
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
            "sensors": [
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "LeftCamera",
                    "socket": "CameraLeftSocket",
                    "Hz": 10,
                    "configuration": {
                        "CaptureWidth": 512,
                        "CaptureHeight": 512,
                        "ExposureMethod": "AEM_Histogram",
                    }
                },
                {
                    "sensor_type": "DynamicsSensor",
                }
            ],
            "control_scheme": 0,
            "location": [0,-10,-12],
            "rotation": [0, 0, 90]

        }
    ]
}

pressed_keys = list()
force = 25

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.remove(key.char)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val*2
    if 'k' in keys:
        command[0:4] -= val*2
    if 'j' in keys:
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys:
        command[[4,7]] -= val
        command[[5,6]] += val

    if 'w' in keys:
        command[4:8] += val*2
    if 's' in keys:
        command[4:8] -= val*2
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val

    return command


# Initialize AprilTag detector
# options = apriltag.DetectorOptions(families="tag36h11")
options = apriltag.DetectorOptions(families="tag16h5")
detector = apriltag.Detector(options)

idx = 0


with holoocean.make(scenario_cfg=cfg) as env:
    env.weather.set_fog_density(1)
    
    while True:
        if 'q' in pressed_keys:
            break
        command = parse_keys(pressed_keys, force)
        env.act("auv0", command)
        simul_state = env.tick()
        
         ## Get the AUV state from the simulation
        auv_state = get_states_6dof(simul_state["DynamicsSensor"])
         
        pose = auv_state["pose"]
        vel = auv_state["velocity"]
        t = auv_state["time"]
        
        
        
        if "LeftCamera" in simul_state:
            image = simul_state["LeftCamera"]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the grayscale image
            results = detector.detect(gray)
            # print(results)
            for r in results:
                # Extract the bounding box (x, y)-coordinates for the AprilTag
                (ptA, ptB, ptC, ptD) = r.corners
                ptA = (int(ptA[0]), int(ptA[1]))
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))

                # Draw the bounding box of the AprilTag detection
                cv2.line(image, ptA, ptB, (0, 255, 0), 2)
                cv2.line(image, ptB, ptC, (0, 255, 0), 2)
                cv2.line(image, ptC, ptD, (0, 255, 0), 2)
                cv2.line(image, ptD, ptA, (0, 255, 0), 2)

                # Draw the center (x, y)-coordinates of the AprilTag
                (cX, cY) = (int(r.center[0]), int(r.center[1]))
                cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

                # Draw the tag family and ID on the image
                tag_family = r.tag_family.decode("utf-8") if hasattr(r.tag_family, 'decode') else r.tag_family
                tag_id = r.tag_id
                text = f"{tag_family} ID:{tag_id}"
                cv2.putText(image, text, (ptA[0], ptA[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if idx % 100 ==0:
                    print(pose)
                    print(f"X: {pose[0]}, Y: {pose[1]}, Z: {pose[2]}, psi: {pose[5]}")
                    # Print tag info
                    print(f"[INFO] Detected tag family: {tag_family}, ID: {tag_id}")
                    idx+=1
                else:
                    idx = 0
                

            cv2.imshow("Camera Feed", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
