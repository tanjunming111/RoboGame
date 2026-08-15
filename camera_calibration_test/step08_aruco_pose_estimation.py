# step08_aruco_pose_estimation.py
# 标定第8步：通过单个ArUco标记（ID=0）实时估计相机位姿
# ============================================================
# 原理：
#   1. 相机检测到ArUco标记（ID=0）→ 得到4个角点2D坐标
#   2. 根据标记已知尺寸，生成3D角点坐标（以标记中心为原点）
#   3. solvePnP求相机相对标记的位姿(rvec, tvec)
#   4. 直接输出相机在标记坐标系下的位置和姿态
#
# 单标记测试模式：
#   - 只检测ID=0
#   - 不做多标记加权平均，直接输出solvePnP结果
#   - 画面上画出3D坐标轴，直观验证位姿是否正确
# ============================================================

import cv2
import numpy as np
import json
import os
import time
import math

# ===== ArUco 参数 =====
ARUCO_DICT = cv2.aruco.DICT_4X4_50
TARGET_MARKER_ID = 0  # 只测试ID=0


def load_camera_params():
    """读取标定参数"""
    params_file = "camera_params.json"
    if not os.path.exists(params_file):
        print(f"[错误] 找不到标定文件 {params_file}")
        print(f"       请先运行 step03_zhang_calibration.py 完成标定")
        return None, None

    with open(params_file, 'r', encoding='utf-8') as f:
        params = json.load(f)

    K = np.array(params['camera_matrix'], dtype=np.float64)
    D = np.array(params['dist_coeffs'], dtype=np.float64).flatten()
    print(f"[标定参数] fx={K[0,0]:.2f} fy={K[1,1]:.2f} RMS={params['rms_error']:.4f}")
    return K, D


def load_marker_config():
    """读取标记配置"""
    config_file = "marker_positions.json"
    if not os.path.exists(config_file):
        print(f"[错误] 找不到标记配置 {config_file}")
        print(f"       请先运行 step07_place_aruco_markers.py")
        return None, None

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    marker_size = config['marker_size_mm']
    print(f"[标记配置] 边长={marker_size}mm, 目标ID={TARGET_MARKER_ID}")
    return config, marker_size


def make_marker_3d_points(size):
    """
    生成标记的4个角点3D坐标（以标记中心为原点）

    ArUco角点顺序：左上→右上→右下→左下
    标记平面在 Z=0 上：
      左上 = (-size/2, -size/2, 0)
      右上 = (+size/2, -size/2, 0)
      右下 = (+size/2, +size/2, 0)
      左下 = (-size/2, +size/2, 0)
    """
    half = size / 2.0
    return np.array([
        [-half, -half, 0],  # 左上
        [ half, -half, 0],  # 右上
        [ half,  half, 0],  # 右下
        [-half,  half, 0],  # 左下
    ], dtype=np.float64)


def rvec_tvec_to_matrix(rvec, tvec):
    """将旋转向量+平移向量转换为4×4变换矩阵"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def matrix_to_euler(T):
    """从4×4变换矩阵提取欧拉角（度）"""
    R = T[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = 0
    return pitch, yaw, roll


def main():
    # ===== 读取标定参数 =====
    K, D = load_camera_params()
    if K is None:
        return

    # ===== 读取标记配置 =====
    config, marker_size = load_marker_config()
    if config is None:
        return

    # 生成标记3D角点（以标记中心为原点）
    obj_pts = make_marker_3d_points(marker_size)

    print("\n" + "=" * 60)
    print("标定第8步：单标记位姿估计（ID=0）")
    print("=" * 60)
    print()
    print("原理：")
    print("  1. 检测ArUco标记ID=0 → 2D角点")
    print("  2. 用标记已知尺寸生成3D角点")
    print("  3. solvePnP求相机相对标记的位姿")
    print("  4. 直接输出相机位置和欧拉角")
    print()
    print("画面说明：")
    print("  检测到标记画绿色框 + ID号")
    print("  标记上画3D坐标轴（红=X, 绿=Y, 蓝=Z）")
    print("  右上角显示相机位置（mm）和欧拉角（度）")
    print("  底部显示重投影误差和距离")
    print()
    print("按键：[q]退出  [s]截图")
    print("=" * 60)

    # ===== 初始化ArUco检测器 =====
    # 新版 OpenCV（4.7+）使用 ArucoDetector 类替代旧的 detectMarkers 函数
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    # ===== 打开摄像头 =====
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 畸变校正
        frame_undist = cv2.undistort(frame, K, D)

        # ===== 1. 检测ArUco标记 =====
        # 新版 OpenCV 用 detector.detectMarkers() 替代 cv2.aruco.detectMarkers()
        corners, ids, rejected = detector.detectMarkers(frame_undist)

        display = frame_undist.copy()

        if ids is not None and TARGET_MARKER_ID in ids:
            # 找到目标标记的索引
            idx = list(ids.ravel()).index(TARGET_MARKER_ID)

            # 画检测框（手动绘制4个角点连线，兼容所有OpenCV版本）
            pts = corners[idx].reshape(-1, 2).astype(int)
            for k in range(4):
                p1 = tuple(pts[k])
                p2 = tuple(pts[(k + 1) % 4])
                cv2.line(display, p1, p2, (0, 255, 0), 2)
            cv2.putText(display, f"ID={TARGET_MARKER_ID}",
                        (int(pts[0][0]), int(pts[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 获取2D角点
            img_pts = corners[idx].reshape(-1, 2).astype(np.float64)

            # ===== 2. solvePnP求位姿 =====
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, None,
                flags=cv2.SOLVEPNP_EPNP
            )

            if success:
                # 用迭代法细化结果
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, None,
                    rvec=rvec, tvec=tvec,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                # ===== 3. 计算重投影误差 =====
                reproj, _ = cv2.projectPoints(obj_pts.astype(np.float32),
                                               rvec, tvec, K, None)
                reproj_error = np.mean(np.linalg.norm(
                    img_pts - reproj.reshape(-1, 2), axis=1))

                # ===== 4. 提取位姿信息 =====
                # solvePnP返回的 rvec, tvec 含义：
                #   将点从标记坐标系变换到相机坐标系的变换 [R|t]
                #   即 tvec = 标记原点在相机坐标系下的位置
                #
                # 我们需要的是：相机在标记坐标系下的位置
                #   T_cam_to_marker = [R | t]（solvePnP求出，标记→相机）
                #   T_marker_to_cam = inv(T_cam_to_marker)（相机→标记）
                #   取逆后的平移部分 = 相机在标记坐标系下的位置
                T_cm = rvec_tvec_to_matrix(rvec, tvec)  # 标记→相机的变换
                T_mc = np.linalg.inv(T_cm)               # 相机→标记的变换（取逆）

                # 相机在标记坐标系下的位置
                cam_pos = T_mc[:3, 3]
                distance = np.linalg.norm(cam_pos)

                # 用逆矩阵的旋转部分求欧拉角（相机在标记坐标系下的姿态）
                pitch, yaw, roll = matrix_to_euler(T_mc)

                # ===== 5. 在标记上画3D坐标轴 =====
                # 画坐标轴可以直观看出位姿是否正确
                # 红色=X轴, 绿色=Y轴, 蓝色=Z轴（朝向相机）
                axis_length = float(marker_size) * 0.8  # 坐标轴长度
                axis_3d = np.array([
                    [0, 0, 0],              # 原点
                    [axis_length, 0, 0],    # X轴端点
                    [0, axis_length, 0],    # Y轴端点
                    [0, 0, -axis_length],   # Z轴端点（负=朝向相机）
                ], dtype=np.float64)
                axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, None)
                axis_2d = axis_2d.reshape(-1, 2).astype(int)

                # 画三条坐标轴线
                origin = tuple(axis_2d[0])
                cv2.line(display, origin, tuple(axis_2d[1]), (0, 0, 255), 3)  # X=红
                cv2.line(display, origin, tuple(axis_2d[2]), (0, 255, 0), 3)  # Y=绿
                cv2.line(display, origin, tuple(axis_2d[3]), (255, 0, 0), 3)  # Z=蓝
                # 标注轴名
                cv2.putText(display, "X", tuple(axis_2d[1] + [5, -5]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(display, "Y", tuple(axis_2d[2] + [5, -5]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display, "Z", tuple(axis_2d[3] + [5, -5]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # ===== 6. 显示位姿信息 =====
                info_lines = [
                    f"Marker: ID={TARGET_MARKER_ID}",
                    f"Pos: X={cam_pos[0]:.1f} Y={cam_pos[1]:.1f} Z={cam_pos[2]:.1f} mm",
                    f"Dist: {distance:.1f} mm",
                    f"Rot: P={pitch:.1f} Y={yaw:.1f} R={roll:.1f} deg",
                    f"Reproj err: {reproj_error:.2f} px",
                ]
                for j, line in enumerate(info_lines):
                    cv2.putText(display, line,
                                (display.shape[1] - 310, 25 + j * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)
        else:
            cv2.putText(display, f"Looking for ID={TARGET_MARKER_ID}...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        cv2.imshow("ArUco Single Marker Pose", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"aruco_pose_{ts}.png", display)
            print(f"[截图] 已保存 (时间戳: {ts})")

    cap.release()
    cv2.destroyAllWindows()
    print("\n单标记位姿估计完成！")


if __name__ == "__main__":
    main()
