# step13_aruco_id5_pose_estimation.py
# 标定第13步：识别 ArUco.png 中的标记 ID=5，实时估计相机位姿
# ============================================================
# ArUco.png 内容说明（已实际检测验证）：
#   - 使用的字典是 AprilTag 36h11（cv2.aruco.DICT_APRILTAG_36H11）
#   - 共6个标记（ID=1~6），2行3列排布：
#       [ID=1] [ID=2] [ID=3]
#       [ID=4] [ID=5] [ID=6]
#
# 原理（与step08单标记模式相同）：
#   1. 检测标记ID=5 → 得到4个角点2D坐标
#   2. 根据标记已知尺寸生成3D角点坐标（以标记中心为原点）
#   3. solvePnP求相机相对标记的位姿(rvec, tvec)
#   4. 取逆变换得到相机在标记坐标系下的位置和姿态
#
# 外部调用方法（本文件核心接口）：
#   from step09_aruco_id1_pose_estimation import get_camera_pose
#
#   result = get_camera_pose(frame)   # frame: BGR图像(np.ndarray)
#   # 或者不传frame，让函数自己临时打开摄像头拍一帧：
#   result = get_camera_pose()
#
#   result['is_detected']     # bool，是否检测到ID=5
#   result['position_mm']     # np.array([x,y,z])，相机位置(mm)
#   result['euler_deg']       # (pitch,yaw,roll)，相机欧拉角(度)
#   result['distance_mm']     # 相机到标记距离(mm)
#   result['reproj_error_px'] # 重投影误差(px)
# ============================================================

import cv2
import numpy as np
import json
import os
import time
import math

# ===== ArUco 参数 =====
ARUCO_DICT = cv2.aruco.DICT_APRILTAG_36H11  # ArUco.png实际使用的字典（已验证）
TARGET_MARKER_ID = 5                        # 本文件只识别 ID=5
MARKER_SIZE_MM = 56.0                       # 标记打印后的实际边长(mm)，实测后修改！

# ===== 模块级缓存（外部反复调用时避免重复读文件/重复创建检测器） =====
_camera_params_cache = None  # 缓存标定参数 (K, D)
_detector_cache = None       # 缓存ArUco检测器实例


def load_camera_params():
    # """读取标定参数（带缓存：只有第一次真正读文件）"""
    global _camera_params_cache
    if _camera_params_cache is not None:
        return _camera_params_cache

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
    _camera_params_cache = (K, D)
    return K, D


def get_detector():
    """获取ArUco检测器（懒加载+缓存，全局只创建一次）"""
    global _detector_cache
    if _detector_cache is None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        detector_params = cv2.aruco.DetectorParameters()
        _detector_cache = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    return _detector_cache


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


def get_camera_pose(frame=None, camera=None, marker_size_mm=None, K=None, D=None):
    """
    外部调用接口：识别目标标记ID=5，返回相机位姿

    参数：
        frame  : BGR图像(np.ndarray)。为None时从camera读一帧
        camera : 已打开的cv2.VideoCapture对象（frame为None时使用；
                 frame和camera都为None时临时打开摄像头0，用完立即释放）
        marker_size_mm : 标记边长(mm)，为None时用模块常量MARKER_SIZE_MM
        K, D   : 相机内参/畸变系数，为None时自动从camera_params.json读取

    返回 dict：
        is_detected     : bool，是否检测到目标标记
        position_mm     : np.array([x,y,z])，相机在标记坐标系下的位置(mm)，未检测到为None
        euler_deg       : (pitch,yaw,roll)，相机姿态欧拉角(度)，未检测到为None
        distance_mm     : float，相机到标记直线距离(mm)，未检测到为None
        reproj_error_px : float，重投影误差(px)，未检测到为None
        rvec / tvec     : solvePnP原始结果（标记→相机变换），未检测到为None
        corners         : 标记4个角点2D坐标(4,2)，坐标系见detected_on，未检测到为None
        detected_on     : 'undistorted'=畸变校正后图像上检测（标定相机画面，正常情况）
                          'raw'         =原始图像上检测（输入非标定相机图像时的兜底路径）
    """
    # 统一的"未检测到"返回结构
    result = {'is_detected': False, 'position_mm': None, 'euler_deg': None,
              'distance_mm': None, 'reproj_error_px': None,
              'rvec': None, 'tvec': None, 'corners': None,
              'detected_on': None}

    # ===== 0. 获取图像帧 =====
    if frame is None:
        temp_camera = None
        if camera is None:
            temp_camera = cv2.VideoCapture(0)  # 未传入任何图像源→临时打开摄像头
            camera = temp_camera
        ret, frame = camera.read()
        if temp_camera is not None:
            temp_camera.release()  # 临时打开的摄像头用完立即释放
        if not ret:
            print("[警告] 无法获取图像帧")
            return result

    # ===== 1. 读取标定参数（带缓存） =====
    if K is None or D is None:
        K, D = load_camera_params()
        if K is None:
            return result

    size = marker_size_mm if marker_size_mm is not None else MARKER_SIZE_MM

    # ===== 2. 检测标记（双路径，提高鲁棒性） =====
    # 路径A：畸变校正后检测 —— 正常情况（画面来自标定过的相机），与step08一致
    # 路径B：原始图像检测 —— 兜底。当输入图像并非来自标定相机时（例如静态图片
    #        ArUco.png，分辨率与标定时不同），强行校正会破坏图像导致检测失败
    frame_undist = cv2.undistort(frame, K, D)
    corners_all, ids, _ = get_detector().detectMarkers(frame_undist)
    detected_on = 'undistorted'
    solve_D = None  # 校正后图像上的角点无畸变，solvePnP不再传D

    if ids is None or TARGET_MARKER_ID not in ids:
        corners_all, ids, _ = get_detector().detectMarkers(frame)
        detected_on = 'raw'
        # 原始图像通常来自非标定相机（分辨率/畸变与标定不一致），
        # 标定畸变模型不适用，按无畸变求解（实测：静态图重投影误差<0.2px）
        solve_D = None
        if ids is None or TARGET_MARKER_ID not in ids:
            return result  # 两条路径都没看到目标标记 → is_detected=False

    # ===== 3. solvePnP求位姿 =====
    idx = list(ids.ravel()).index(TARGET_MARKER_ID)
    img_pts = corners_all[idx].reshape(-1, 2).astype(np.float64)
    obj_pts = make_marker_3d_points(size)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, solve_D, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        return result
    # 用迭代法细化结果
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, solve_D, rvec=rvec, tvec=tvec,
        flags=cv2.SOLVEPNP_ITERATIVE)

    # ===== 4. 计算重投影误差 =====
    reproj, _ = cv2.projectPoints(obj_pts.astype(np.float32),
                                  rvec, tvec, K, solve_D)
    reproj_error = np.mean(np.linalg.norm(
        img_pts - reproj.reshape(-1, 2), axis=1))

    # ===== 5. 相机在标记坐标系下的位姿 =====
    # solvePnP求的是 标记→相机 的变换 [R|t]，取逆得到 相机→标记 的变换，
    # 逆变换的平移部分 = 相机在标记坐标系下的位置
    T_cm = rvec_tvec_to_matrix(rvec, tvec)  # 标记→相机
    T_mc = np.linalg.inv(T_cm)              # 相机→标记
    cam_pos = T_mc[:3, 3]
    distance = float(np.linalg.norm(cam_pos))
    pitch, yaw, roll = matrix_to_euler(T_mc)

    result.update({
        'is_detected': True,
        'position_mm': cam_pos,
        'euler_deg': (pitch, yaw, roll),
        'distance_mm': distance,
        'reproj_error_px': float(reproj_error),
        'rvec': rvec,
        'tvec': tvec,
        'corners': img_pts,
        'detected_on': detected_on,
    })
    return result


def main():
    # ===== 读取标定参数（main里读一次，供畸变校正使用） =====
    K, D = load_camera_params()
    if K is None:
        return

    print("\n" + "=" * 60)
    print("标定第13步：单标记位姿估计（ArUco.png, ID=5, AprilTag36h11）")
    print("=" * 60)
    print()
    print("原理：")
    print("  1. 检测AprilTag36h11标记ID=5 → 2D角点")
    print("  2. 用标记已知尺寸生成3D角点")
    print("  3. solvePnP求相机相对标记的位姿")
    print("  4. 输出相机位置和欧拉角")
    print()
    print("画面说明：")
    print("  检测到标记画绿色框 + ID号")
    print("  标记上画3D坐标轴（红=X, 绿=Y, 蓝=Z）")
    print("  右上角显示相机位置（mm）和欧拉角（度）")
    print("  底部显示重投影误差和距离")
    print()
    print("外部调用接口：get_camera_pose(frame) → dict（含is_detected）")
    print("按键：[q]退出  [s]截图")
    print("=" * 60)

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

        # 畸变校正（正常情况在校正后画面上检测和显示）
        frame_undist = cv2.undistort(frame, K, D)

        # ===== 调用外部接口获取位姿（传入当前帧） =====
        result = get_camera_pose(frame=frame, K=K, D=D)

        if result['is_detected']:
            # 根据检测路径选择显示底图：
            #   undistorted → 底图用校正后画面
            #   raw         → 底图用原始画面
            if result['detected_on'] == 'undistorted':
                display = frame_undist.copy()
            else:
                display = frame.copy()
            proj_D = None  # 两条路径的solvePnP都按无畸变求解，投影同样不传D
            # 画检测框（手动绘制4个角点连线，兼容所有OpenCV版本）
            pts = result['corners'].astype(int)
            for k in range(4):
                p1 = tuple(pts[k])
                p2 = tuple(pts[(k + 1) % 4])
                cv2.line(display, p1, p2, (0, 255, 0), 2)
            cv2.putText(display, f"ID={TARGET_MARKER_ID}",
                        (int(pts[0][0]), int(pts[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            rvec = result['rvec']
            tvec = result['tvec']
            cam_pos = result['position_mm']
            pitch, yaw, roll = result['euler_deg']

            # ===== 在标记上画3D坐标轴 =====
            # 画坐标轴可以直观看出位姿是否正确
            # 红色=X轴, 绿色=Y轴, 蓝色=Z轴（朝向相机）
            axis_length = MARKER_SIZE_MM * 0.8  # 坐标轴长度
            axis_3d = np.array([
                [0, 0, 0],              # 原点
                [axis_length, 0, 0],    # X轴端点
                [0, axis_length, 0],    # Y轴端点
                [0, 0, -axis_length],   # Z轴端点（负=朝向相机）
            ], dtype=np.float64)
            axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, proj_D)
            axis_2d = axis_2d.reshape(-1, 2).astype(int)

            origin = tuple(axis_2d[0])
            cv2.line(display, origin, tuple(axis_2d[1]), (0, 0, 255), 3)  # X=红
            cv2.line(display, origin, tuple(axis_2d[2]), (0, 255, 0), 3)  # Y=绿
            cv2.line(display, origin, tuple(axis_2d[3]), (255, 0, 0), 3)  # Z=蓝
            cv2.putText(display, "X", tuple(axis_2d[1] + [5, -5]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(display, "Y", tuple(axis_2d[2] + [5, -5]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display, "Z", tuple(axis_2d[3] + [5, -5]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # ===== 显示位姿信息 =====
            info_lines = [
                f"Marker: ID={TARGET_MARKER_ID}",
                f"Pos: X={cam_pos[0]:.1f} Y={cam_pos[1]:.1f} Z={cam_pos[2]:.1f} mm",
                f"Dist: {result['distance_mm']:.1f} mm",
                f"Rot: P={pitch:.1f} Y={yaw:.1f} R={roll:.1f} deg",
                f"Reproj err: {result['reproj_error_px']:.2f} px",
            ]
            for j, line in enumerate(info_lines):
                cv2.putText(display, line,
                            (display.shape[1] - 310, 25 + j * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
        else:
            # 未检测到：默认显示校正后画面
            display = frame_undist.copy()
            cv2.putText(display, f"Looking for ID={TARGET_MARKER_ID}...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        cv2.imshow(f"ArUco Pose ID={TARGET_MARKER_ID}", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"aruco_pose_id{TARGET_MARKER_ID}_{ts}.png", display)
            print(f"[截图] 已保存 (时间戳: {ts})")

    cap.release()
    cv2.destroyAllWindows()
    print("\n单标记位姿估计完成！")


if __name__ == "__main__":
    main()
