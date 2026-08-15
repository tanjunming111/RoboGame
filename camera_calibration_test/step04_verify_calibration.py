# step04_verify_calibration.py
# 标定第4步：验证标定效果 —— 畸变校正对比
# ============================================================
# 原理：
#   用标定得到的 K 和 D 对实时画面做畸变校正（undistort）
#   对比原始画面和校正后画面，直观验证标定质量
#   如果校正后直线变直了 → 标定成功
#   如果变形更严重 → 需重新标定
# 用法：
#   python step04_verify_calibration.py
#   按 [q] 退出，按 [s] 截图
# ============================================================

import cv2
import numpy as np
import json
import os
import time


def nothing(x):
    pass


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

    print("=" * 60)
    print("标定第4步：验证标定效果")
    print("=" * 60)
    print()
    print(f"内参矩阵 K =")
    print(f"  fx = {K[0,0]:.4f}, fy = {K[1,1]:.4f}")
    print(f"  cx = {K[0,2]:.4f}, cy = {K[1,2]:.4f}")
    print(f"  RMS误差 = {params['rms_error']:.4f}")
    print()
    print("画面说明：")
    print("  左 = 原始画面（有畸变）")
    print("  右 = 畸变校正后（直线应该变直）")
    print()
    print("验证方法：")
    print("  1. 拿一个直边物体（如书本边缘）对着相机")
    print("  2. 原始画面中边缘应该是弯的（桶形畸变）")
    print("  3. 校正后边缘应变直")
    print("  4. 如果更弯了说明标定有误，需重做")
    print()
    print("按键：[q]退出  [s]截图")
    print("=" * 60)

    # ===== 打开摄像头 =====
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, params['image_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params['image_height'])

    # 计算去畸变后的有效区域
    # getOptimalNewCameraMatrix 计算 newK，alpha=0 时裁掉黑边，=1 保留全部
    w, h = params['image_width'], params['image_height']
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== 畸变校正 =====
        # 方法1: undistort（直接校正，简单但边缘可能有黑边）
        # 方法2: 先 initUndistortRectifyMap 再 remap（更快，适合实时）
        undistorted = cv2.undistort(frame, K, D, None, newK)

        # 裁掉黑边区域
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y+rh, x:x+rw]
            # 缩放回原尺寸方便对比
            undistorted = cv2.resize(undistorted, (w, h))

        # ===== 拼接对比 =====
        # 左：原始，右：校正后
        combined = np.hstack([frame, undistorted])

        # 画一条参考竖线，方便看直线是否变直
        cv2.line(combined, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(combined, (w + w // 2, 0), (w + w // 2, h), (0, 255, 0), 1)

        # 标签
        cv2.putText(combined, "Original (distorted)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(combined, "Undistorted", (w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 参数信息
        cv2.putText(combined,
                    f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} RMS={params['rms_error']:.3f}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)

        cv2.imshow("Calibration Verification", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"calibration_verify_{ts}.png", combined)
            print(f"[截图] 已保存 (时间戳: {ts})")

    cap.release()
    cv2.destroyAllWindows()
    print("\n下一步：运行 step05_pnp_pose_estimation.py 用标定结果做位姿估计")


if __name__ == "__main__":
    main()
