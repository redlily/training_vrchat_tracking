import math
import threading

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask
from matplotlib import pyplot as plt
from pythonosc import udp_client
from scipy.spatial.transform import Rotation

pose_points = [np.array([0, 0, 0], dtype=np.float32) for i in range(33)]  # カメラ座標系のポーズ座標
pose_world_points = [np.array([0, 0, 0], dtype=np.float32) for i in range(33)]  # 腰を原点としたワールド座情系のポーズ座標
pose_virtual_points = [np.array([0, 0, 0], dtype=np.float32) for i in range(33)]  # 仮想空間座標系のポーズ座標
pose_virtual_rotation = {
    "chest": np.eye(3, dtype=np.float32),
    "hip": np.eye(3, dtype=np.float32),
    "right_knee": np.eye(3, dtype=np.float32),
    "reft_knee": np.eye(3, dtype=np.float32),
    "left_foot": np.eye(3, dtype=np.float32),
    "right_foot": np.eye(3, dtype=np.float32),
}


def update_pose(pose_landmarks, pose_world_landmarks, image_size):
    if pose_landmarks is not None:
        for i in range(33):
            landmark = pose_landmarks.landmark[i]
            world_landmark = pose_world_landmarks.landmark[i]
            if landmark.visibility:
                pose_points[i] = np.array([landmark.x, landmark.y, landmark.z],
                                          dtype=np.float32)
                pose_world_points[i] = np.array([world_landmark.x, world_landmark.y, world_landmark.z],
                                                dtype=np.float32)

    global pose_virtual_points
    if calibration_enabled:
        # カメラから見える腰の大きさからZ値算出
        hip_direction = pose_world_points[23] - pose_world_points[24]
        hip_direction /= np.linalg.norm(hip_direction)
        hip_cos = hip_direction @ np.array([0, 0, 1], dtype=np.float32)
        hip_sin = math.sqrt(1 - hip_cos ** 2)
        hip_depth = (calibration_hip_distance * hip_sin) / np.linalg.norm((pose_points[24] - pose_points[26])[:2])

        # ワールド座標系の座標シフト値を計算
        hip_point = (pose_points[23] + pose_points[24]) / 2
        shift_point = hip_point * calibration_scale

        pose_virtual_points = [calibration_matrix @ np.append(pose_world_points[i], 1.0)
                               for i in range(33)]

        # pose_virtual_points = [calibration_matrix @ np.append(pose_world_points[i] + shift_point, 1.0)
        #                        for i in range(33)]

        # min_y = min(point[1] for point in pose_virtual_points)
        # for i in range(33):
        #     pose_virtual_points[i][1] -= min_y
    else:
        pose_virtual_points = pose_world_points

    # 回転を計算
    right_shoulder_point = pose_virtual_points[12][:3]
    left_shoulder_point = pose_virtual_points[11][:3]
    chest_point = (right_shoulder_point + left_shoulder_point) / 2

    right_hip_point = pose_virtual_points[24][:3]
    left_hip_point = pose_virtual_points[23][:3]
    hip_point = (right_hip_point + left_hip_point) / 2

    right_elbow = pose_virtual_points[14][:3]
    right_knee_point = pose_virtual_points[26][:3]
    right_ankle_point = pose_virtual_points[28][:3]
    right_heel_point = pose_virtual_points[30][:3]
    right_foot_index_point = pose_virtual_points[32][:3]

    left_elbow = pose_virtual_points[15][:3]
    left_knee_point = pose_virtual_points[25][:3]
    left_ankle_point = pose_virtual_points[27][:3]
    left_heel_point = pose_virtual_points[29][:3]
    left_foot_index_point = pose_virtual_points[31][:3]

    # 胸の回転を計算
    chest_axis_x = right_shoulder_point - left_shoulder_point
    chest_axis_x /= np.linalg.norm(chest_axis_x)
    chest_axis_y = chest_point - hip_point
    chest_axis_y /= np.linalg.norm(chest_axis_y)
    chest_axis_z = np.cross(chest_axis_x, chest_axis_y)
    pose_virtual_rotation["chest"] = np.array([chest_axis_x, chest_axis_y, chest_axis_z], dtype=np.float32).T

    # 腰の回転を計算
    hip_axis_x = right_hip_point - left_hip_point
    hip_axis_x /= np.linalg.norm(hip_axis_x)
    hip_axis_y = chest_point - hip_point
    hip_axis_y /= np.linalg.norm(hip_axis_y)
    hip_axis_z = np.cross(hip_axis_x, hip_axis_y)
    pose_virtual_rotation["hip"] = np.array([hip_axis_x, hip_axis_y, hip_axis_z], dtype=np.float32).T

    # 右股の回転を計算
    right_knee_axis_y = right_hip_point - right_knee_point
    right_knee_axis_y /= np.linalg.norm(right_knee_axis_y)
    right_knee_axis_ny = right_ankle_point - right_knee_point
    right_knee_axis_ny /= np.linalg.norm(right_knee_axis_ny)
    right_knee_axis_y_cos = right_knee_axis_y @ right_knee_axis_ny
    if right_knee_axis_y_cos < -0.9:
        # 膝がほぼ伸び切っている状態の場合、かかととつま先のベクトルと股のベクトルの外積をX軸として代替
        right_knee_axis_x = np.cross(right_knee_axis_y, right_foot_index_point - right_heel_point)
        right_knee_axis_x /= np.linalg.norm(right_knee_axis_x)
    else:
        # 膝がある程度曲がっている場外
        right_knee_axis_x = np.cross(right_knee_axis_ny, right_knee_axis_y)
        right_knee_axis_x /= np.linalg.norm(right_knee_axis_x)
    right_knee_axis_z = np.cross(right_knee_axis_x, right_knee_axis_y)
    pose_virtual_rotation["right_knee"] = np.array([right_knee_axis_x, right_knee_axis_y, right_knee_axis_z],
                                                   dtype=np.float32).T

    # 左股の回転を計算
    left_knee_axis_y = left_hip_point - left_knee_point
    left_knee_axis_y /= np.linalg.norm(left_knee_axis_y)
    left_knee_axis_ny = left_ankle_point - left_knee_point
    left_knee_axis_ny /= np.linalg.norm(left_knee_axis_ny)
    left_knee_axis_y_cos = left_knee_axis_y @ left_knee_axis_ny
    if left_knee_axis_y_cos < -0.9:
        # 膝がほぼ伸び切っている状態の場合、かかととつま先のベクトルと股のベクトルの外積をX軸として代替
        left_knee_axis_x = np.cross(left_knee_axis_y, left_foot_index_point - left_heel_point)
        left_knee_axis_x /= np.linalg.norm(left_knee_axis_x)
    else:
        # 膝がある程度曲がっている場外
        left_knee_axis_x = np.cross(left_knee_axis_ny, left_knee_axis_y)
        left_knee_axis_x /= np.linalg.norm(left_knee_axis_x)
    left_knee_axis_z = np.cross(left_knee_axis_x, left_knee_axis_y)
    pose_virtual_rotation["left_knee"] = np.array([left_knee_axis_x, left_knee_axis_y, left_knee_axis_z],
                                                   dtype=np.float32).T

    # 右足首の回転を計算
    right_foot_axis_z = right_foot_index_point - right_heel_point
    right_foot_axis_z /= np.linalg.norm(right_foot_axis_z)
    right_foot_axis_x = np.cross(right_knee_point - right_heel_point, right_foot_axis_z)
    right_foot_axis_x /= np.linalg.norm(right_foot_axis_x)
    right_foot_axis_y = np.cross(right_foot_axis_z, right_foot_axis_x)
    pose_virtual_rotation["right_foot"] = np.array([right_foot_axis_x, right_foot_axis_y, right_foot_axis_z],
                                                   dtype=np.float32).T

    # 左足首の回転を計算
    left_foot_axis_z = left_foot_index_point - left_heel_point
    left_foot_axis_z /= np.linalg.norm(left_foot_axis_z)
    left_foot_axis_x = np.cross(left_knee_point - left_heel_point, left_foot_axis_z)
    left_foot_axis_x /= np.linalg.norm(left_foot_axis_x)
    left_foot_axis_y = np.cross(left_foot_axis_z, left_foot_axis_x)
    pose_virtual_rotation["left_foot"] = np.array([left_foot_axis_x, left_foot_axis_y, left_foot_axis_z],
                                                  dtype=np.float32).T

    # 右腕の回転を計算
    right_elbow_axis_x = right_ankle_point - right_knee_point

    # 左腕の回転を計算


calibration_enabled = False
calibration_matrix = np.eye(4, dtype=np.float32)
calibration_hip_distance = 1  # Z深度の基準となる腰の長さ
calibration_scale = 1  # pose_landmarkとpose_world_landmarksの変換倍率


def update_calibration_parameter():
    global calibration_enabled
    calibration_enabled = True

    # Z深度の基準となる腰の長さを保存
    global calibration_hip_distance
    calibration_hip_distance = np.linalg.norm((pose_points[23] - pose_points[24])[:2])

    # pose_landmarkとpose_world_landmarksの変換倍率を保存
    global calibration_scale
    calibration_scale = (np.linalg.norm((pose_world_points[23] - pose_world_points[24])[:2]) /
                         np.linalg.norm((pose_points[23] - pose_points[24])[:2]))

    # 平行移動の補正値算出
    hip_point = (pose_points[23] + pose_points[24]) / 2
    world_top_point = (pose_world_points[11] + pose_world_points[12]) / 2
    world_bottom_point = (pose_world_points[29] + pose_world_points[30]) / 2

    translation_mat = np.eye(4, dtype=np.float32)
    translation_mat[:3, 3] = -world_bottom_point - hip_point * calibration_scale

    # Y軸傾きの補正値算出
    y_axis = np.array([0, 1, 0], dtype=np.float32)
    y_slop = (world_top_point - world_bottom_point) / np.linalg.norm(world_top_point - world_bottom_point)
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

    # サイズ補正
    height = np.linalg.norm(pose_world_points[20] - pose_world_points[19])
    scale_mat = np.eye(4, dtype=np.float32)
    scale_mat[0, 0] = 1.75 * 1 / height
    scale_mat[1, 1] = 1.75 * 1 / height
    scale_mat[2, 2] = 1.75 * 1 / height

    # 座標系反転
    modify_coordination_system_mat = np.eye(4, dtype=np.float32)
    modify_coordination_system_mat[0, 0] = -1

    global calibration_matrix
    calibration_matrix = modify_coordination_system_mat @ y_slop_mat @ translation_mat


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

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

def update_plot():
    while True:
        ax.cla()

        if calibration_enabled:
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(0, 2)
        else:
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(1, -1)

        for group in LANDMARK_GROUPS:
            x = [pose_virtual_points[i][0] for i in group]
            y = [pose_virtual_points[i][1] for i in group]
            z = [pose_virtual_points[i][2] for i in group]

            ax.plot(x, z, y)

        plt.pause(0.05)


vr_pose = {
    # left foot
    28: {
        "position_address": "/tracking/trackers/1/position",
        "rotation_address": "/tracking/trackers/1/rotation"
    },
    # right foot
    27: {
        "position_address": "/tracking/trackers/2/position",
        "rotation_address": "/tracking/trackers/2/rotation"
    },
    # left knee
    26: {
        "position_address": "/tracking/trackers/3/position",
        "rotation_address": "/tracking/trackers/3/rotation"
    },
    # right knee
    25: {
        "position_address": "/tracking/trackers/4/position",
        "rotation_address": "/tracking/trackers/4/rotation"
    },
    # left hip
    24: {
        "position_address": "/tracking/trackers/5/position",
        "rotation_address": "/tracking/trackers/5/rotation"
    },
    # right hip
    23: {
        "position_address": "/tracking/trackers/5/position",
        "rotation_address": "/tracking/trackers/5/rotation"
    },
    # left chest
    12: {
        "position_address": "/tracking/trackers/6/position",
        "rotation_address": "/tracking/trackers/6/rotation"
    },
    # right chest
    11: {
        "position_address": "/tracking/trackers/6/position",
        "rotation_address": "/tracking/trackers/6/rotation"
    },
    # left elbow
    14: {
        "position_address": "/tracking/trackers/7/position",
        "rotation_address": "/tracking/trackers/7/rotation"
    },
    # right elbow
    13: {
        "position_address": "/tracking/trackers/8/position",
        "rotation_address": "/tracking/trackers/8/rotation"
    },
}

vrchat_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

rot_test = 0


def send_pose_to_vrchat():
    # 右足
    vrchat_client.send_message(vr_pose[28]["position_address"], pose_virtual_points[28][:3].tolist())
    right_foot_rotation = Rotation.from_matrix(pose_virtual_rotation["right_foot"])
    right_foot_angle = right_foot_rotation.as_euler("zxy", degrees=True).tolist()
    vrchat_client.send_message(vr_pose[28]["rotation_address"],
                               [right_foot_angle[1], right_foot_angle[2], right_foot_angle[0]])

    # 右股
    vrchat_client.send_message(vr_pose[26]["position_address"],
                               ((pose_virtual_points[26][:3] * 6 + pose_virtual_points[24][:3]) / 7).tolist())
    right_knee_rotation = Rotation.from_matrix(pose_virtual_rotation["right_knee"])
    right_knee_angle = right_knee_rotation.as_euler("zxy", degrees=True).tolist()
    vrchat_client.send_message(vr_pose[26]["rotation_address"],
                               [right_knee_angle[1], right_knee_angle[2], right_knee_angle[0]])

    # 左足
    vrchat_client.send_message(vr_pose[27]["position_address"], pose_virtual_points[27][:3].tolist())
    left_foot_rotation = Rotation.from_matrix(pose_virtual_rotation["left_foot"])
    left_foot_angle = left_foot_rotation.as_euler("zxy", degrees=True).tolist()
    vrchat_client.send_message(vr_pose[27]["rotation_address"],
                               [left_foot_angle[1], left_foot_angle[2], left_foot_angle[0]])

    # 左股
    vrchat_client.send_message(vr_pose[25]["position_address"],
                               ((pose_virtual_points[25][:3] * 6 + pose_virtual_points[23][:3]) / 7).tolist())
    left_knee_rotation = Rotation.from_matrix(pose_virtual_rotation["left_knee"])
    left_knee_angle = left_knee_rotation.as_euler("zxy", degrees=True).tolist()
    vrchat_client.send_message(vr_pose[25]["rotation_address"],
                               [left_knee_angle[1], left_knee_angle[2], left_knee_angle[0]])

    # 腰
    hip = (pose_virtual_points[24][:3] + pose_virtual_points[23][:3]) / 2
    vrchat_client.send_message(vr_pose[24]["position_address"], hip.tolist())
    hip_rotation = Rotation.from_matrix(pose_virtual_rotation["hip"])
    hip_angle = hip_rotation.as_euler("zxy", degrees=True).tolist()
    vrchat_client.send_message(vr_pose[24]["rotation_address"], [hip_angle[1], hip_angle[2], hip_angle[0]])

    # 胸
    chest = (pose_virtual_points[11][:3] + pose_virtual_points[12][:3]) / 2
    vrchat_client.send_message(vr_pose[11]["position_address"], ((chest * 5 + hip) / 6).tolist())
    chest_rotation = Rotation.from_matrix(pose_virtual_rotation["chest"])
    chest_angle = chest_rotation.as_euler("zxy", degrees=True).tolist()
    vrchat_client.send_message(vr_pose[11]["rotation_address"], [chest_angle[1], chest_angle[2], chest_angle[0]])

    # 腕
    vrchat_client.send_message(vr_pose[14]["position_address"],
                               ((pose_virtual_points[14][:3] * 5 + pose_virtual_points[12][:3]) / 6).tolist())
    vrchat_client.send_message(vr_pose[13]["position_address"],
                               ((pose_virtual_points[13][:3] * 5 + pose_virtual_points[11][:3]) / 6).tolist())

    # 頭
    vrchat_client.send_message("/tracking/trackers/head/position", chest.tolist())


web_app = Flask(__name__)


@web_app.route("/calibration")
def web_calibration():
    update_calibration_parameter()
    return "Success"


def run_flask():
    web_app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    threading.Thread(target=run_analyze_pose).start()
    threading.Thread(target=run_flask).start()
    update_plot()
