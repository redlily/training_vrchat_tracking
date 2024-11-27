import threading

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, make_response
from pythonosc import udp_client

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# Initialize OSC Client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

# For webcam input:
vr_pose = {
    # left foot
    28: {
        "address": "/tracking/trackers/1/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
    # right foot
    27: {
        "address": "/tracking/trackers/2/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
    # left knee
    26: {
        "address": "/tracking/trackers/3/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
    # right knee
    25: {
        "address": "/tracking/trackers/4/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
    # left hip
    24: {
        "address": "/tracking/trackers/5/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
    # right hip
    23: {
        "address": "/tracking/trackers/5/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
    # left chest
    12: {
        "address": "/tracking/trackers/6/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
    # right chest
    11: {
        "address": "/tracking/trackers/6/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
    # left elbow
    14: {
        "address": "/tracking/trackers/7/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
    # right elbow
    13: {
        "address": "/tracking/trackers/8/position",
        "coordination": [0.0, 0.0, 0.0],
        "offset": 0.0
    },
}

# Init Flask
web_app = Flask(__name__)

@web_app.route('/z_offset')
def z_offset():
    for key in vr_pose.keys():
        vr_pose[key]["offset"] = vr_pose[key]["coordination"][2]
    return make_response("ok", 200)

def web_task():
    web_app.run(host='localhost', port=8000)
web_thread = threading.Thread(target=web_task)
web_thread.start()

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_world_landmarks:
        for index, pose_landmark in enumerate(results.pose_landmarks.landmark):
            if pose_landmark.visibility:
                if index == 16:
                    print(f"{pose_landmark.x}, {pose_landmark.y}, {pose_landmark.z}")
                v_pose = vr_pose.get(index)
                if v_pose :
                    v_pose["coordination"][0] = (pose_landmark.y * 1.0) * 1.8
                    v_pose["coordination"][1] = (-pose_landmark.x * 1.0 + 1.0) * 1.8
                    v_pose["coordination"][2] = (-pose_landmark.z * 1.0) * 1.0 #- v_pose["offset"]

        client.send_message(vr_pose[28]["address"], vr_pose[28]["coordination"])
        client.send_message(vr_pose[27]["address"], vr_pose[27]["coordination"])
        client.send_message(vr_pose[26]["address"], vr_pose[26]["coordination"])
        client.send_message(vr_pose[25]["address"], vr_pose[25]["coordination"])
        
        client.send_message(vr_pose[24]["address"],
                            [(vr_pose[24]["coordination"][0] + vr_pose[23]["coordination"][0]) / 2,
                             (vr_pose[24]["coordination"][1] + vr_pose[23]["coordination"][1]) / 2,
                             (vr_pose[24]["coordination"][2] + vr_pose[23]["coordination"][2]) / 2])
        client.send_message(vr_pose[12]["address"],
                            [(vr_pose[12]["coordination"][0] + vr_pose[11]["coordination"][0]) / 2,
                             (vr_pose[12]["coordination"][1] + vr_pose[11]["coordination"][1]) / 2,
                             (vr_pose[12]["coordination"][2] + vr_pose[11]["coordination"][2]) / 2])
        
        client.send_message(vr_pose[14]["address"], vr_pose[14]["coordination"])
        client.send_message(vr_pose[13]["address"], vr_pose[13]["coordination"])
        
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
