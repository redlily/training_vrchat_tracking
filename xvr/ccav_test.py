import cv2
import numpy as np

world_points = np.array([
    [0, 0, 0],
    [-0.38, 0, 0],
    [0, 0.15, 0],
    [0, 0, -0.28],
    [-0.38, 0.15, 0],
    [0, 0.15, -0.28],
    [-0.38, 0.15, -0.28],
    [0, 0.15, -0.28 / 2],
    [-0.38, 0.15, -0.28 / 2],
], dtype=np.float32)

image_points = np.array([
    [593, 2928],
    [1248, 2790],
    [498, 2694],
    [590, 2688],
    [1249, 2571],
    [489, 2483],
    [1079, 2387],
    [535, 2576],
    [1152, 2465]
], dtype=np.float32)

def compute_camera_matrix(world_points, image_points):
    # 行列 A の構築
    A = []
    for X, Y, Z, u, v in zip(world_points[:, 0], world_points[:, 1], world_points[:, 2], image_points[:, 0], image_points[:, 1]):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    A = np.array(A)

    # 特異値分解 (SVD)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)  # 最後の行が解
    return P

P = compute_camera_matrix(world_points, image_points)
print("Camera Matrix:\n", P)

for i in range(world_points.shape[0]):
    p = np.dot(P, np.append(world_points[i], 1))
    print(f"{p[0] / p[2]}, {p[1] / p[2]}")

# カメラ行列 P を分解
K = P[:, :3]  # カメラの内部パラメータ行列 (3x3)
R_t = np.linalg.inv(K) @ P[:, 3]  # 回転行列と並進ベクトルを求める

# カメラの位置を計算
# カメラの位置（ワールド座標系における位置）
camera_position = -R_t

print("カメラの位置:", camera_position)
