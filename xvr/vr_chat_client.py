import threading

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask
from matplotlib import pyplot as plt
from pythonosc import udp_client
from scipy.spatial.transform import Rotation

# カメラ座標系のポーズ座標
pose_points = [np.array([0, 0, 0], dtype=np.float32) for i in range(33)]

# 腰を原点としたワールド座情系のポーズ座標
pose_world_points = [np.array([0, 0, 0], dtype=np.float32) for i in range(33)]

# 補正されたVRC用の座標系のポーズ座標
pose_virtual_points = [np.array([0, 0, 0], dtype=np.float32) for i in range(33)]

# VRC用の座標変換の数値
pose_virtual_transforms = {
    "head": {
        "path": "head",
        "enable": True,
        "position": np.array([0, 0, 0], dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    },
    "chest": {
        "path": "1",
        "enable": True,
        "position": np.array([0, 0, 0], dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    },
    "hip": {
        "path": "2",
        "enable": True,
        "position": np.array([0, 0, 0], dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    },
    "right_elbow": {
        "path": "3",
        "enable": True,
        "position": np.array([0, 0, 0], dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    },
    "left_elbow": {
        "path": "4",
        "enable": True,
        "position": np.array([0, 0, 0], dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    },
    "right_knee": {
        "path": "5",
        "enable": True,
        "position": np.array([0, 0, 0], dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    },
    "left_knee": {
        "path": "6",
        "enable": True,
        "position": np.array([0, 0, 0], dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    },
    "left_foot": {
        "path": "7",
        "enable": True,
        "position": np.array([0, 0, 0], dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    },
    "right_foot": {
        "path": "8",
        "enable": True,
        "position": np.array([0, 0, 0], dtype=np.float32),
        "rotation": np.eye(3, dtype=np.float32),
    }
}


def update_pose(pose_landmarks, pose_world_landmarks, image_size):
    """
    ポーズの更新を行う
    """
    if pose_landmarks is not None:
        for i in range(33):
            landmark = pose_landmarks.landmark[i]
            world_landmark = pose_world_landmarks.landmark[i]
            if landmark.visibility:
                pose_points[i] = np.array(
                    [landmark.x - 0.5, (landmark.y - 0.5) * (image_size[1] / image_size[0]), landmark.z],
                    dtype=np.float32)
                pose_world_points[i] = np.array([world_landmark.x, world_landmark.y, world_landmark.z],
                                                dtype=np.float32)

    global pose_virtual_points
    if calibration_enabled:
        pose_virtual_points = [calibration_matrix @ np.append(pose_world_points[i], 1.0) for i in range(33)]
        modify_virtual_pose()
    else:
        pose_virtual_points = pose_world_points

    update_virtual_pose()

def modify_virtual_pose():
    """
    VRC用のポーズの補正を行う
    """

    # 胴体の長さの補正
    chest = (pose_virtual_points[11] + pose_virtual_points[12]) / 2
    hip = (pose_virtual_points[23] + pose_virtual_points[24]) / 2
    body_vector = hip - chest
    body_length = np.linalg.norm(body_vector)
    body_differential_length = calibration_body_length - body_length
    body_modify_length = (body_vector / body_length) * body_differential_length
    for i in range(23, 33):
        pose_virtual_points[i] += body_modify_length

def update_virtual_pose():
    """
    VRC用のポーズの更新を行う
    """
    eye_position = (pose_virtual_points[8][:3] + pose_virtual_points[7][:3]) / 2

    right_shoulder_position = pose_virtual_points[12][:3]
    left_shoulder_position = pose_virtual_points[11][:3]
    chest_position = (right_shoulder_position + left_shoulder_position) / 2

    right_hip_position = pose_virtual_points[24][:3]
    left_hip_position = pose_virtual_points[23][:3]
    hip_position = (right_hip_position + left_hip_position) / 2

    right_elbow_position = pose_virtual_points[14][:3]
    right_wrist_position = pose_virtual_points[16][:3]
    right_knee_position = pose_virtual_points[26][:3]
    right_ankle_position = pose_virtual_points[28][:3]
    right_heel_position = pose_virtual_points[30][:3]
    right_foot_index_position = pose_virtual_points[32][:3]

    left_elbow_position = pose_virtual_points[13][:3]
    left_wrist_position = pose_virtual_points[15][:3]
    left_knee_position = pose_virtual_points[25][:3]
    left_ankle_position = pose_virtual_points[27][:3]
    left_heel_position = pose_virtual_points[29][:3]
    left_foot_index_position = pose_virtual_points[31][:3]

    # 頭の回転を計算
    head_axis_y = eye_position - chest_position
    head_axis_y /= np.linalg.norm(head_axis_y)
    head_axis_x = right_shoulder_position - left_shoulder_position
    head_axis_z = np.cross(head_axis_x, head_axis_y)
    head_axis_z /= np.linalg.norm(head_axis_z)
    head_axis_x = np.cross(head_axis_y, head_axis_z)
    pose_virtual_transforms["head"]["rotation"] = np.array([head_axis_x, head_axis_y, head_axis_z], dtype=np.float32).T
    pose_virtual_transforms["head"]["position"] = eye_position

    # 胸の回転を計算
    chest_axis_x = right_shoulder_position - left_shoulder_position
    chest_axis_x /= np.linalg.norm(chest_axis_x)
    chest_axis_y = chest_position - hip_position
    chest_axis_z = np.cross(chest_axis_x, chest_axis_y)
    chest_axis_z /= np.linalg.norm(chest_axis_z)
    chest_axis_y = np.cross(chest_axis_z, chest_axis_x)
    pose_virtual_transforms["chest"]["rotation"] = np.array([chest_axis_x, chest_axis_y, chest_axis_z],
                                                            dtype=np.float32).T
    pose_virtual_transforms["chest"]["position"] = chest_position

    # 腰の回転を計算
    hip_axis_x = right_hip_position - left_hip_position
    hip_axis_x /= np.linalg.norm(hip_axis_x)
    hip_axis_y = chest_position - hip_position
    hip_axis_z = np.cross(hip_axis_x, hip_axis_y)
    hip_axis_z /= np.linalg.norm(hip_axis_z)
    hip_axis_y = np.cross(hip_axis_z, hip_axis_x)
    pose_virtual_transforms["hip"]["rotation"] = np.array([hip_axis_x, hip_axis_y, hip_axis_z], dtype=np.float32).T
    pose_virtual_transforms["hip"]["position"] = hip_position

    # 右肘の回転を計算
    right_elbow_axis_x = right_elbow_position - right_shoulder_position
    right_elbow_axis_x /= np.linalg.norm(right_elbow_axis_x)
    right_wrist_axis_x = right_wrist_position - right_elbow_position
    right_wrist_axis_x /= np.linalg.norm(right_wrist_axis_x)
    if right_elbow_axis_x @ right_wrist_axis_x > 0.9:
        # 肘がほとんど伸び切っている場合、手のひらの向きを腕のY軸ベクトルとして代替
        right_elbow_axis_y = np.cross(pose_virtual_points[20][:3], pose_virtual_points[18][:3])
    else:
        # 肘が曲がっている場合
        right_elbow_axis_y = np.cross(right_wrist_axis_x, right_elbow_axis_x)
    right_elbow_axis_z = np.cross(right_elbow_axis_x, right_elbow_axis_y)
    right_elbow_axis_z /= np.linalg.norm(right_elbow_axis_z)
    right_elbow_axis_y = np.cross(right_elbow_axis_z, right_elbow_axis_x)
    pose_virtual_transforms["right_elbow"]["rotation"] = np.array(
        [right_elbow_axis_x, right_elbow_axis_y, right_elbow_axis_z], dtype=np.float32).T
    pose_virtual_transforms["right_elbow"]["position"] = (right_elbow_position * 3 + right_shoulder_position) / 4

    # 左肘の回転を計算
    left_elbow_axis_x = left_shoulder_position - left_elbow_position
    left_elbow_axis_x /= np.linalg.norm(left_elbow_axis_x)
    left_wrist_axis_x = left_elbow_position - left_wrist_position
    left_wrist_axis_x /= np.linalg.norm(left_wrist_axis_x)
    if left_elbow_axis_x @ left_wrist_axis_x > 0.9:
        # 肘がほとんど伸び切っている場合、手のひらの向きを腕のY軸ベクトルとして代替
        left_elbow_axis_y = np.cross(pose_virtual_points[17][:3], pose_virtual_points[19][:3])
    else:
        # 肘が曲がっている場合
        left_elbow_axis_y = np.cross(left_elbow_axis_x, left_wrist_axis_x)
    left_elbow_axis_z = np.cross(left_elbow_axis_x, left_elbow_axis_y)
    left_elbow_axis_z /= np.linalg.norm(left_elbow_axis_z)
    left_elbow_axis_y = np.cross(left_elbow_axis_z, left_elbow_axis_x)
    pose_virtual_transforms["left_elbow"]["rotation"] = np.array(
        [left_elbow_axis_x, left_elbow_axis_y, left_elbow_axis_z], dtype=np.float32).T
    pose_virtual_transforms["left_elbow"]["position"] = (left_elbow_position * 3 + left_shoulder_position) / 4

    # 右足首の回転を計算
    right_foot_axis_z = right_foot_index_position - right_heel_position
    right_foot_axis_z /= np.linalg.norm(right_foot_axis_z)
    right_foot_axis_y = right_knee_position - right_heel_position
    right_foot_axis_x = np.cross(right_foot_axis_y, right_foot_axis_z)
    right_foot_axis_x /= np.linalg.norm(right_foot_axis_x)
    right_foot_axis_y = np.cross(right_foot_axis_z, right_foot_axis_x)
    pose_virtual_transforms["right_foot"]["rotation"] = np.array(
        [right_foot_axis_x, right_foot_axis_y, right_foot_axis_z], dtype=np.float32).T
    pose_virtual_transforms["right_foot"]["position"] = right_ankle_position

    # 右足首の回転を計算
    left_foot_axis_z = left_foot_index_position - left_heel_position
    left_foot_axis_z /= np.linalg.norm(left_foot_axis_z)
    left_foot_axis_y = left_knee_position - left_heel_position
    left_foot_axis_x = np.cross(left_foot_axis_y, left_foot_axis_z)
    left_foot_axis_x /= np.linalg.norm(left_foot_axis_x)
    left_foot_axis_y = np.cross(left_foot_axis_z, left_foot_axis_x)
    pose_virtual_transforms["left_foot"]["rotation"] = np.array(
        [left_foot_axis_x, left_foot_axis_y, left_foot_axis_z], dtype=np.float32).T
    pose_virtual_transforms["left_foot"]["position"] = left_ankle_position

    # 右膝の回転を計算
    right_knee_axis_y = right_hip_position - right_knee_position
    right_knee_axis_y /= np.linalg.norm(right_knee_axis_y)
    right_ankle_axis_y = right_knee_position - right_ankle_position
    right_ankle_axis_y /= np.linalg.norm(right_ankle_axis_y)
    if right_knee_axis_y @ right_ankle_axis_y > 0.9:
        # 膝がほとんど伸び切っている場合、つま先の向きからX軸ベクトルを算出
        right_knee_axis_x = right_foot_axis_x
    else:
        # 膝が曲がっている場合
        right_knee_axis_x = np.cross(right_knee_axis_y, right_ankle_axis_y)
        right_knee_axis_x /= np.linalg.norm(right_knee_axis_x)
    right_knee_axis_z = np.cross(right_knee_axis_x, right_knee_axis_y)
    pose_virtual_transforms["right_knee"]["rotation"] = np.array(
        [right_knee_axis_x, right_knee_axis_y, right_knee_axis_z], dtype=np.float32).T
    pose_virtual_transforms["right_knee"]["position"] = right_knee_position

    # 左膝の回転を計算
    left_knee_axis_y = left_hip_position - left_knee_position
    left_knee_axis_y /= np.linalg.norm(left_knee_axis_y)
    left_ankle_axis_y = left_knee_position - left_ankle_position
    left_ankle_axis_y /= np.linalg.norm(left_ankle_axis_y)
    if left_knee_axis_y @ left_ankle_axis_y > 0.9:
        # 膝がほとんど伸び切っている場合、つま先の向きからX軸ベクトルを算出
        left_knee_axis_x = left_foot_axis_x
    else:
        # 膝が曲がっている場合
        left_knee_axis_x = np.cross(left_knee_axis_y, left_ankle_axis_y)
        left_knee_axis_x /= np.linalg.norm(left_knee_axis_x)
    left_knee_axis_z = np.cross(left_knee_axis_x, left_knee_axis_y)
    pose_virtual_transforms["left_knee"]["rotation"] = np.array(
        [left_knee_axis_x, left_knee_axis_y, left_knee_axis_z], dtype=np.float32).T
    pose_virtual_transforms["left_knee"]["position"] = left_knee_position


# キャリブレーションが行われているかどうか
calibration_enabled = False

# 座標補正の行列
calibration_matrix = np.eye(4, dtype=np.float32)

# 胸から腰までの長さ
calibration_body_length = 1.0


def update_calibration_parameter():
    global calibration_enabled
    calibration_enabled = True

    # 体の部位の計測
    global calibration_body_length
    calibration_body_length = np.linalg.norm(
        (pose_world_points[11] + pose_world_points[12]) / 2 - (pose_world_points[23] + pose_world_points[24]) / 2)

    top_point = (pose_world_points[7] + pose_world_points[8]) / 2
    bottom_point = (pose_world_points[29] + pose_world_points[30]) / 2

    # Y軸傾きの補正値算出
    y_axis = np.array([0, 1, 0], dtype=np.float32)
    y_slop = (top_point - bottom_point) / np.linalg.norm(top_point - bottom_point)
    y_slop_cos = y_axis @ y_slop
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

    # 座標系変換
    modify_coordination_system_mat = np.eye(4, dtype=np.float32)
    modify_coordination_system_mat[0, 0] = -1

    global calibration_matrix
    calibration_matrix = modify_coordination_system_mat @ y_slop_mat


def run_analyze_pose():
    mp_pose = mp.solutions.pose
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

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks and results.pose_world_landmarks:
                update_pose(results.pose_landmarks, results.pose_world_landmarks, (image.shape[0], image.shape[1]))
                send_pose_to_vrchat()

        cap.release()


# VRChat用のOSCクライアント
vrchat_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)


def send_pose_to_vrchat():
    """
    VRChatにポーズを送信する
    """
    for key in pose_virtual_transforms:
        value = pose_virtual_transforms[key]
        if value["enable"]:
            position = value["position"]
            rotation_mat = value["rotation"]
            rotation_rot = Rotation.from_matrix(rotation_mat)
            rotation_zxy = rotation_rot.as_euler("zxy", degrees=True)
            vrchat_client.send_message(f"/tracking/trackers/{value["path"]}/position", position.tolist())
            vrchat_client.send_message(f"/tracking/trackers/{value["path"]}/rotation",
                                       [rotation_zxy[1], rotation_zxy[2], rotation_zxy[0]])


web_app = Flask(__name__)


@web_app.route("/calibration")
def web_calibration():
    update_calibration_parameter()
    return "Success"


def run_flask():
    web_app.run(debug=True, use_reloader=False)


# ランドマークのグループ
landmark_groups = [
    [8, 6, 5, 4, 0, 1, 2, 3, 7],  # 目
    [10, 9],  # 口
    [11, 13, 15, 17, 19, 15, 21],  # 右腕
    [11, 23, 25, 27, 29, 31, 27],  # 右半身
    [12, 14, 16, 18, 20, 16, 22],  # 左腕
    [12, 24, 26, 28, 30, 32, 28],  # 左半身
    [11, 12],  # 肩
    [23, 24],  # 腰
]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot_axis = False


def update_plot():
    while True:
        ax.cla()

        if calibration_enabled:
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
        else:
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(1, -1)

        for group in landmark_groups:
            x = [pose_virtual_points[i][0] for i in group]
            y = [pose_virtual_points[i][1] for i in group]
            z = [pose_virtual_points[i][2] for i in group]

            ax.plot(x, z, y)

        if plot_axis:
            for key in pose_virtual_transforms:
                transform = pose_virtual_transforms[key]
                position = transform["position"]
                rotation = transform["rotation"]
                axis_x = position + rotation[:, 0] * 0.1
                axis_y = position + rotation[:, 1] * 0.1
                axis_z = position + rotation[:, 2] * 0.1

                xx = [position[0], axis_x[0]]
                xy = [position[1], axis_x[1]]
                xz = [position[2], axis_x[2]]
                ax.plot(xx, xz, xy, color="r")

                yx = [position[0], axis_y[0]]
                yy = [position[1], axis_y[1]]
                yz = [position[2], axis_y[2]]
                ax.plot(yx, yz, yy, color="g")

                zx = [position[0], axis_z[0]]
                zy = [position[1], axis_z[1]]
                zz = [position[2], axis_z[2]]
                ax.plot(zx, zz, zy, color="b")

        plt.pause(0.05)


if __name__ == '__main__':
    threading.Thread(target=run_analyze_pose).start()
    threading.Thread(target=run_flask).start()
    update_plot()
