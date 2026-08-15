# step05_pnp_pose_estimation.py
# 标定第5步：用标定结果做 PnP 位姿估计
# ============================================================
# 原理：
#   标定后得到了相机内参 K 和畸变 D
#   现在拿着棋盘格对着相机，用 PnP 算法实时求解相机相对棋盘格的位姿
#   输出：旋转向量 rvec → 旋转矩阵 R → 欧拉角（俯仰/偏航/翻滚）
#         平移向量 tvec → 相机在棋盘格坐标系下的 (x, y, z) 位置
#
# PnP 流程：
#   1. 检测棋盘格角点（2D像素坐标）
#   2. 已知角点的3D坐标（棋盘格平面上）
#   3. 用 solvePnP 求 rvec, tvec
#   4. Rodrigues 转旋转矩阵 → 欧拉角
#   5. 可视化位姿（画3D坐标轴）
# 用法：
#   python step05_pnp_pose_estimation.py
#   按 [q] 退出，按 [s] 截图
# ============================================================

import cv2
import numpy as np
import json
import os
import time
import math


def nothing(x):
    pass


def rotation_vector_to_euler(rvec):
    """
    将旋转向量转换为欧拉角（俯仰pitch、偏航yaw、翻滚roll）

    原理：
      rvec 是罗德里格斯旋转向量，长度=旋转角度，方向=旋转轴
      先用 Rodrigues 转成 3×3 旋转矩阵 R
      再从 R 分解出三个轴的旋转角度

    返回：(pitch, yaw, roll) 单位：度
      pitch = 绕 X 轴旋转（俯仰）
      yaw   = 绕 Y 轴旋转（偏航）
      roll  = 绕 Z 轴旋转（翻滚）
    """
    R, _ = cv2.Rodrigues(rvec)

    # 从旋转矩阵分解欧拉角（ZYX顺序）
    # 注意：OpenCV 相机坐标系 z 轴朝前
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
    params_file = "camera_params.json"
    if not os.path.exists(params_file):
        print(f"[错误] 找不到标定文件 {params_file}")
        print(f"       请先运行 step03_zhang_calibration.py 完成标定")
        return

    with open(params_file, 'r') as f:
        params = json.load(f)

    K = np.array(params['camera_matrix'], dtype=np.float64)
    D = np.array(params['dist_coeffs'], dtype=np.float64)

    CHESSBOARD_COLS = params['chessboard_cols']
    CHESSBOARD_ROWS = params['chessboard_rows']
    SQUARE_SIZE = params['square_size_mm']

    print("=" * 60)
    print("标定第5步：PnP 位姿估计")
    print("=" * 60)
    print()
    print(f"内参：fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"      cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    print(f"棋盘格：{CHESSBOARD_COLS}×{CHESSBOARD_ROWS}, 方格={SQUARE_SIZE}mm")
    print()
    print("原理：")
    print("  1. 检测棋盘格角点 → 2D像素坐标")
    print("  2. 已知角点3D坐标（棋盘格平面）")
    print("  3. solvePnP 求 rvec(旋转), tvec(平移)")
    print("  4. Rodrigues 转旋转矩阵 → 欧拉角")
    print()
    print("画面说明：")
    print("  棋盘格上画3D坐标轴：红=X, 绿=Y, 蓝=Z(朝相机)")
    print("  右侧显示位姿数值（位置mm + 欧拉角度）")
    print()
    print("按键：[q]退出  [s]截图")
    print("=" * 60)

    # ===== 构建3D角点坐标 =====
    # 与 step03 一致：棋盘格放在 Z=0 平面
    objp = np.zeros((CHESSBOARD_COLS * CHESSBOARD_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # 定义3D坐标轴的端点（用于画坐标轴）
    # 原点在棋盘格第一个角点，三轴各延伸3个方格长度
    axis_length = SQUARE_SIZE * 3
    axis_3d = np.float32([
        [0, 0, 0],                  # 原点
        [axis_length, 0, 0],         # X 轴端点（红）
        [0, axis_length, 0],        # Y 轴端点（绿）
        [0, 0, -axis_length],       # Z 轴端点（蓝，朝向相机）
    ]).reshape(-1, 3)

    # ===== 打开摄像头 =====
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, params['image_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params['image_height'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 先做畸变校正（标定后每帧都应先 undistort）
        frame_undist = cv2.undistort(frame, K, D)
        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

        # ===== 1. 检测棋盘格角点 =====
        found, corners = cv2.findChessboardCorners(
            gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = frame_undist.copy()

        if found:
            # 亚像素精细化
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11),
                                                (-1, -1), criteria)

            # ===== 2. solvePnP 求位姿 =====
            # 输入：3D点 + 2D点 + 内参K + 畸变D
            # 输出：rvec(旋转向量), tvec(平移向量，单位mm)
            success, rvec, tvec = cv2.solvePnP(
                objp, corners_refined, K, None,  # 畸变已校正，D传None
                flags=cv2.SOLVEPNP_EPNP
            )

            if success:
                # 画角点
                cv2.drawChessboardCorners(display,
                                          (CHESSBOARD_COLS, CHESSBOARD_ROWS),
                                          corners_refined, found)

                # ===== 3. 投影3D坐标轴到图像 =====
                # projectPoints：用 rvec, tvec, K 把3D点投影到2D
                axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, None)

                # 画坐标轴
                origin = tuple(axis_2d[0].ravel().astype(int))
                pt_x = tuple(axis_2d[1].ravel().astype(int))
                pt_y = tuple(axis_2d[2].ravel().astype(int))
                pt_z = tuple(axis_2d[3].ravel().astype(int))

                cv2.line(display, origin, pt_x, (0, 0, 255), 3)   # X红
                cv2.line(display, origin, pt_y, (0, 255, 0), 3)   # Y绿
                cv2.line(display, origin, pt_z, (255, 0, 0), 3)   # Z蓝

                # ===== 4. 计算欧拉角 =====
                pitch, yaw, roll = rotation_vector_to_euler(rvec)

                # ===== 5. 显示位姿信息 =====
                # tvec = 相机在棋盘格坐标系下的位置
                # 负号转换：棋盘格在相机坐标系下的位置
                tx, ty, tz = tvec.ravel()

                info_lines = [
                    f"Position (mm): X={tx:.1f}  Y={ty:.1f}  Z={tz:.1f}",
                    f"Distance: {math.sqrt(tx**2+ty**2+tz**2):.1f} mm",
                    f"Rotation: Pitch={pitch:.1f}  Yaw={yaw:.1f}  Roll={roll:.1f}",
                ]

                y_offset = 30
                for line in info_lines:
                    cv2.putText(display, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)
                    y_offset += 22

                # 画一个小窗口显示距离
                cv2.putText(display, f"Z={tz:.0f}mm",
                            (display.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

        else:
            cv2.putText(display, "Show chessboard to camera",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        cv2.imshow("PnP Pose Estimation", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"pnp_pose_{ts}.png", display)
            print(f"[截图] 已保存 (时间戳: {ts})")

    cap.release()
    cv2.destroyAllWindows()
    print("\n标定流程全部完成！")
    print("标定结果保存在 camera_params.json，可以在其他项目中复用")


if __name__ == "__main__":
    main()
