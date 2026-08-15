# step03_zhang_calibration.py
# 标定第3步：张正友标定法 —— 求相机内参和畸变系数
# ============================================================
# 张正友标定法原理（2000年提出）：
#   1. 假设标定板是平面（Z=0），建立已知3D坐标
#   2. 从不同角度拍摄多张照片，每张提取2D角点
#   3. 利用单应性矩阵（Homography）建立3D→2D的映射
#   4. 约束条件：内参矩阵满足 |r1|=|r2| 且 r1·r2=0（旋转矩阵正交性）
#   5. 求解线性方程组得到内参矩阵 K
#   6. 用最大似然估计（Levenberg-Marquardt）非线性优化 refine
#   7. 顺便得到每张图片的外参（R, t）和畸变系数（k1~k5）
#
# OpenCV 的 calibrateCamera() 内部实现了上述完整流程
# ============================================================

import cv2
import numpy as np
import os
import json
import glob

# ===== 参数 =====
CHESSBOARD_COLS = 9    # 棋盘格内角点列数
CHESSBOARD_ROWS = 6    # 棋盘格内角点行数
SQUARE_SIZE = 28.0     # 方格实际边长（毫米），打印后用尺子测量！
IMAGE_DIR = "calibration_images"  # 标定图片目录
OUTPUT_FILE = "camera_params.json"  # 输出文件


def main():
    print("=" * 60)
    print("标定第3步：张正友标定法求内参")
    print("=" * 60)

    # ===== 1. 构建3D角点坐标 =====
    # 棋盘格放在 Z=0 平面上，角点坐标为 (x, y, 0)
    # x = 列号 × 方格边长，y = 行号 × 方格边长
    # 单位：毫米
    objp = np.zeros((CHESSBOARD_COLS * CHESSBOARD_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # 乘以实际边长

    # 存储所有图片的3D点和2D点
    objpoints = []  # 3D 点（在标定板坐标系下）
    imgpoints = []  # 2D 点（在图像像素坐标系下）
    image_shape = None

    # ===== 2. 读取所有标定图片，检测角点 =====
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    if len(images) == 0:
        print(f"[错误] 在 {IMAGE_DIR}/ 目录下没有找到标定图片")
        print(f"       请先运行 step02_capture_calibration.py 拍摄图片")
        return

    print(f"\n找到 {len(images)} 张标定图片")
    print(f"方格边长：{SQUARE_SIZE} mm")
    print(f"内角点数：{CHESSBOARD_COLS}×{CHESSBOARD_ROWS} = {CHESSBOARD_COLS * CHESSBOARD_ROWS} 个\n")
    print("开始检测角点...")
    print("-" * 50)

    success_count = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]  # (width, height)

        # 检测棋盘格角点
        found, corners = cv2.findChessboardCorners(
            gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # 亚像素精细化角点位置
            # 窗口大小 (11,11)：在角点附近 11×11 区域内搜索精确位置
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11),
                                                (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners_refined)
            success_count += 1
            print(f"  [{idx+1:2d}/{len(images)}] OK   {os.path.basename(fname)}  角点数={len(corners_refined)}")
        else:
            print(f"  [{idx+1:2d}/{len(images)}] FAIL {os.path.basename(fname)}  未检测到角点")

    print("-" * 50)
    print(f"\n成功检测：{success_count}/{len(images)} 张")

    if success_count < 5:
        print("[错误] 成功检测的图片太少（至少需要5张），请重新拍摄")
        return

    # ===== 3. 张正友标定：calibrateCamera =====
    # 参数说明：
    #   objpoints:  每张图片的3D角点坐标
    #   imgpoints:  每张图片的2D角点坐标（像素）
    #   image_shape: 图片尺寸 (width, height)
    #   K:          相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    #   D:          畸变系数 [k1, k2, p1, p2, k3]
    #                k1,k2,k3 = 径向畸变（桶形/枕形）
    #                p1,p2    = 切向畸变（薄棱镜）
    #   rvecs:      每张图片的旋转向量
    #   tvecs:      每张图片的平移向量
    print("\n开始张正友标定（calibrateCamera）...")

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    # 新版 OpenCV 返回的 D 可能是一维数组 (N,) 或二维 (N,1)，统一展平为一维
    D_flat = D.flatten()

    # ===== 4. 输出标定结果 =====
    print("\n" + "=" * 50)
    print("标定结果")
    print("=" * 50)

    print(f"\n重投影误差（RMS）：{ret:.4f} 像素")
    print(f"  → < 0.5 = 优秀")
    print(f"  → 0.5~1.0 = 良好")
    print(f"  → > 1.5 = 需重新标定")

    print(f"\n内参矩阵 K =")
    print(f"  fx = {K[0,0]:.4f}")
    print(f"  fy = {K[1,1]:.4f}")
    print(f"  cx = {K[0,2]:.4f}  （主点x，应接近图像宽度/2）")
    print(f"  cy = {K[1,2]:.4f}  （主点y，应接近图像高度/2）")

    # 畸变系数：前5个为 [k1, k2, p1, p2, k3]
    print(f"\n畸变系数 D（共{len(D_flat)}个，前5个常用）")
    if len(D_flat) >= 1:
        print(f"  k1 = {D_flat[0]:.6f}  （径向畸变1）")
    if len(D_flat) >= 2:
        print(f"  k2 = {D_flat[1]:.6f}  （径向畸变2）")
    if len(D_flat) >= 3:
        print(f"  p1 = {D_flat[2]:.6f}  （切向畸变1）")
    if len(D_flat) >= 4:
        print(f"  p2 = {D_flat[3]:.6f}  （切向畸变2）")
    if len(D_flat) >= 5:
        print(f"  k3 = {D_flat[4]:.6f}  （径向畸变3）")

    # ===== 5. 计算每张图片的重投影误差 =====
    print(f"\n各图片重投影误差：")
    total_error = 0
    for i in range(len(objpoints)):
        # 用标定结果重新投影3D点到2D
        imgpoints_reproj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, D
        )
        # 计算与实际检测到的2D点的误差
        error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / \
                len(imgpoints_reproj)
        total_error += error
        print(f"  图片 {i+1:2d}: {error:.4f} px")

    avg_error = total_error / len(objpoints)
    print(f"\n平均重投影误差：{avg_error:.4f} px")

    # ===== 6. 保存标定结果到 JSON =====
    params = {
        'image_width': image_shape[0],
        'image_height': image_shape[1],
        'chessboard_cols': CHESSBOARD_COLS,
        'chessboard_rows': CHESSBOARD_ROWS,
        'square_size_mm': SQUARE_SIZE,
        'rms_error': ret,
        'avg_reprojection_error': avg_error,
        'camera_matrix': K.tolist(),
        'dist_coeffs': D.tolist(),
        'fx': K[0, 0],
        'fy': K[1, 1],
        'cx': K[0, 2],
        'cy': K[1, 2],
        'num_images': success_count,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print(f"\n[保存] 标定结果已保存到 {OUTPUT_FILE}")
    print("=" * 50)
    print("下一步：运行 step04_verify_calibration.py 验证标定效果")
    print("       运行 step05_pnp_pose_estimation.py 用标定结果做位姿估计")


if __name__ == "__main__":
    main()
