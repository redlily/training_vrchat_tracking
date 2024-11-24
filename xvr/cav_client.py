import threading

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from flask import Flask, request
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client

# ボディー用の座標処理
pose_world_points = [np.array([0, 0, 0], dtype=np.float32) for i in range(33)]
pose_points = [np.array([0, 0, 0], dtype=np.float32) for i in range(33)]


def update_body_points(landmarks, world_landmarks, image_size):
    for i in range(33):
        if landmarks:
            landmark = landmarks[i]
            if landmark.visibility:
                pose_points[i][:] = [landmark.x, landmark.y, landmark.z]

        if world_landmarks:
            landmark = world_landmarks.landmark[i]
            if landmark.visibility:
                pose_world_points[i][:] = [landmark.x, landmark.y, landmark.z]


# ボディーキャリブレーション用の行列更新
calibration_matrix = np.eye(4, dtype=np.float32)
calibration_depth_index = np.float32(1.0)


def update_calibration_parameter():
    # 腰の長さを保存する
    chest = (pose_world_points[11] + pose_world_points[12]) / 2
    hip = (pose_world_points[23] + pose_world_points[24]) / 2
    global calibration_depth_index
    calibration_depth_index = np.linalg.norm(chest - hip)

    #
    # T字ポーズをもとにVRChatの座標系に変換するための
    #
    top = (pose_world_points[11] + pose_world_points[12]) / 2
    bottom = (pose_world_points[29] + pose_world_points[30]) / 2

    # 平行移動の補正値算出
    translation_mat = np.eye(4, dtype=np.float32)
    translation_mat[:3, 3] = -bottom

    # Y軸傾きの補正値算出
    y_axis = np.array([0, 1, 0])
    y_slop = (top - bottom) / np.linalg.norm(top - bottom)
    y_slop_cos = np.dot(y_axis, y_slop)
    y_slop_axis = np.cross(y_slop, y_axis)
    y_slop_sin = np.linalg.norm(y_slop_axis)
    y_slop_axis /= y_slop_sin
    ys_x, ys_y, ys_z = y_slop_axis
    ys_c = y_slop_cos
    ys_s = y_slop_sin
    ys_t = 1.0 - ys_c
    y_slop_mat = np.eye(4, dtype=np.float32)
    y_slop_mat[:3, :3] = np.array([
        [ys_t * ys_x * ys_x + ys_c, ys_t * ys_x * ys_y - ys_s * ys_z, ys_t * ys_x * ys_z + ys_s * ys_y],
        [ys_t * ys_x * ys_y + ys_s * ys_z, ys_t * ys_y * ys_y + ys_c, ys_t * ys_y * ys_z - ys_s * ys_x],
        [ys_t * ys_x * ys_z - ys_s * ys_y, ys_t * ys_y * ys_z + ys_s * ys_x, ys_t * ys_z * ys_z + ys_c]
    ], dtype=np.float32)

    # Z軸正対の補正値算出
    correcting_mat = y_slop_mat @ translation_mat
    corrected_bottom = correcting_mat @ np.append(bottom, 1.0)
    corrected_right_hand = correcting_mat @ np.append(pose_world_points[14], 1.0)
    corrected_left_hand = correcting_mat @ np.append(pose_world_points[13], 1.0)
    z_slop = np.cross((corrected_left_hand[:3] - corrected_bottom[:3]),
                      (corrected_right_hand[:3] - corrected_bottom[:3]))
    z_slop[1] = 0
    z_slop /= np.linalg.norm(z_slop)
    z_slop_mat = np.eye(4, dtype=np.float32)
    z_slop_mat[:3, :3] = np.array([
        [z_slop[2], 0, -z_slop[0]],
        [0, 1, 0],
        [z_slop[0], 0, z_slop[2]]
    ], dtype=np.float32)

    # サイズ補正
    height = np.linalg.norm(top - bottom)
    scale_mat = np.eye(4, dtype=np.float32)
    scale_mat[0, 0] = 1.5 / height
    scale_mat[1, 1] = 1.5 / height
    scale_mat[2, 2] = 1.5 / height

    # 座標系反転
    modify_coordination_system_mat = np.eye(4, dtype=np.float32)
    modify_coordination_system_mat[0, 0] = -1

    global calibration_matrix
    calibration_matrix = modify_coordination_system_mat @ scale_mat @ z_slop_mat @ y_slop_mat @ translation_mat


LANDMARK_GROUPS = [
    [8, 6, 5, 4, 0, 1, 2, 3, 7],  # eyes
    [10, 9],  # mouth
    [11, 13, 15, 17, 19, 15, 21],  # right arm
    [11, 23, 25, 27, 29, 31, 27],  # right body side
    [12, 14, 16, 18, 20, 16, 22],  # left arm
    [12, 24, 26, 28, 30, 32, 28],  # left body side
    [11, 12],  # shoulder
    [23, 24],  # waist
]


def plot_pose():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.cla()

    # had to flip the z axis
    ax.set_xlim3d(1, -1)
    ax.set_ylim3d(1, -1)
    ax.set_zlim3d(0, 2)

    chest = (pose_world_points[11] + pose_world_points[12]) / 2
    hip = (pose_world_points[23] + pose_world_points[24]) / 2
    depth_index = np.linalg.norm(chest - hip)
    depth = calibration_depth_index / depth_index

    corrected_position = [calibration_matrix @
                          np.array([pose_world_points[i][0],
                                    pose_world_points[i][1],
                                    pose_world_points[i][2] + depth,
                                    1.0], dtype=np.float32) for i in range(33)]

    # get coordinates for each group and plot
    for group in LANDMARK_GROUPS:
        x = [corrected_position[i][0] for i in group]
        y = [corrected_position[i][1] for i in group]
        z = [corrected_position[i][2] for i in group]

        ax.plot(x, z, y)

    plt.show()


web_app = Flask(__name__)


@web_app.route("/calibration")
def web_calibration():
    update_calibration_parameter()
    return "Success"


@web_app.route("/plot_pose")
def web_plot_pose():
    calibrate = request.args.get("calibrate", default=1)
    if calibrate == "1":
        update_calibration_parameter()
    plot_pose()
    return "Success"


def run_flask():
    web_app.run(debug=True, use_reloader=False)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

vr_pose = {
    # left foot
    28: {
        "address": "/tracking/trackers/1/position",
    },
    # right foot
    27: {
        "address": "/tracking/trackers/2/position",
    },
    # left knee
    26: {
        "address": "/tracking/trackers/3/position",
    },
    # right knee
    25: {
        "address": "/tracking/trackers/4/position",
    },
    # left hip
    24: {
        "address": "/tracking/trackers/5/position",
    },
    # right hip
    23: {
        "address": "/tracking/trackers/5/position",
    },
    # left chest
    12: {
        "address": "/tracking/trackers/6/position",
    },
    # right chest
    11: {
        "address": "/tracking/trackers/6/position",
    },
    # left elbow
    14: {
        "address": "/tracking/trackers/7/position",
    },
    # right elbow
    13: {
        "address": "/tracking/trackers/8/position",
    },
}

vrchat_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)


def send_pose_to_vrchat():
    chest = (pose_world_points[11] + pose_world_points[12]) / 2
    hip = (pose_world_points[23] + pose_world_points[24]) / 2
    depth_index = np.linalg.norm(chest - hip)
    depth = calibration_depth_index / depth_index

    corrected_position = [calibration_matrix @
                          np.array([pose_world_points[i][0],
                                    pose_world_points[i][1],
                                    pose_world_points[i][2],
                                    1.0], dtype=np.float32) for i in range(33)]

    vrchat_client.send_message("/debug/depth", depth.item())

    # 足
    vrchat_client.send_message(vr_pose[28]["address"],
                               ((corrected_position[28][:3] * 8 + corrected_position[26][:3]) / 9).tolist())
    vrchat_client.send_message(vr_pose[27]["address"],
                               ((corrected_position[27][:3] * 8 + corrected_position[25][:3]) / 9).tolist())
    vrchat_client.send_message(vr_pose[26]["address"],
                               ((corrected_position[26][:3] * 6 + corrected_position[24][:3]) / 7).tolist())
    vrchat_client.send_message(vr_pose[25]["address"],
                               ((corrected_position[25][:3] * 6 + corrected_position[23][:3]) / 7).tolist())

    # 胴
    hip = (corrected_position[24][:3] + corrected_position[23][:3]) / 2
    vrchat_client.send_message(vr_pose[24]["address"], hip.tolist())
    chest = (corrected_position[11][:3] + corrected_position[12][:3]) / 2
    vrchat_client.send_message(vr_pose[11]["address"], ((chest * 5 + hip) / 6).tolist())

    # 腕
    vrchat_client.send_message(vr_pose[14]["address"],
                               ((corrected_position[14][:3] + corrected_position[12][:3]) / 2).tolist())
    vrchat_client.send_message(vr_pose[13]["address"],
                               ((corrected_position[13][:3] + corrected_position[11][:3]) / 2).tolist())


def run_osc_server():
    dis = dispatcher.Dispatcher()
    dis.map("/*", print)

    server = osc_server.ThreadingOSCUDPServer(
        ("localhost", 9001), dis)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()


if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    osc_server_thread = threading.Thread(target=run_osc_server)
    osc_server_thread.start()

# For webcam input:
cap = cv2.VideoCapture(1)
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
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks:
            update_body_points(results.pose_landmarks, results.pose_world_landmarks, (image.shape[0], image.shape[1]))
            send_pose_to_vrchat()

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
