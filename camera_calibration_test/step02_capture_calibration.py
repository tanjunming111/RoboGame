# step02_capture_calibration.py
# 标定第2步：拍摄标定图片
# ============================================================
# 原理：
#   张正友标定法需要从不同角度拍摄棋盘格
#   每张图片提取角点后，用多张图片联合求解内参
#   拍摄要求：
#     1. 至少 15~30 张不同角度的照片
#     2. 覆盖画面的不同区域（左上/右下/中心都要有）
#     3. 角度多样：俯视/仰视/左右倾斜
#     4. 棋盘格要占画面的 30%~80%
#     5. 光照均匀，对焦清晰
# 用法：
#   python step02_capture_calibration.py
#   按 [空格] 拍照保存，按 [q] 退出
# ============================================================

import cv2
import os
import time

# ===== 参数 =====
# 棋盘格内角点数（与 step01 一致）
CHESSBOARD_COLS = 9
CHESSBOARD_ROWS = 6

# 保存目录
SAVE_DIR = "calibration_images"

# ===== 创建保存目录 =====
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"[信息] 创建目录：{SAVE_DIR}/")

def main():
    # ===== 打开摄像头 =====
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 统计已有图片数
    existing = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.png')])
    count = existing

    print("=" * 60)
    print("标定第2步：拍摄标定图片")
    print("=" * 60)
    print()
    print(f"棋盘格内角点：{CHESSBOARD_COLS}×{CHESSBOARD_ROWS}")
    print(f"保存目录：{SAVE_DIR}/")
    print(f"已有图片：{count} 张")
    print()
    print("拍摄要求：")
    print("  - 至少拍 15~30 张")
    print("  - 不同角度：俯视/仰视/左右倾斜")
    print("  - 不同位置：棋盘格出现在画面各处")
    print("  - 棋盘格占画面 30%~80%")
    print("  - 对焦清晰，光照均匀")
    print()
    print("画面说明：")
    print("  绿框 = 检测到棋盘格角点（可以拍照）")
    print("  红字 = 未检测到（调整角度或距离）")
    print()
    print("按键：[空格]拍照  [q]退出  [d]删除最后一张")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ===== 实时检测棋盘格角点 =====
        # findChessboardCorners 返回是否找到和角点坐标
        found, corners = cv2.findChessboardCorners(
            gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # 绘制检测结果
        display = frame.copy()
        if found:
            # 精细化角点位置（亚像素精度）
            # criteria：终止条件，最多迭代30次或精度达到0.001
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                criteria)

            # 画绿色框 + 角点
            cv2.drawChessboardCorners(display, (CHESSBOARD_COLS, CHESSBOARD_ROWS),
                                      corners_refined, found)
            cv2.putText(display, "DETECTED - Press SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
        else:
            cv2.putText(display, "NOT DETECTED - Adjust angle",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        # 显示当前已拍数量
        cv2.putText(display, f"Captured: {count}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Capture Calibration Images", display)

        # ===== 按键处理 =====
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格拍照
            if found:
                # 保存图片
                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(SAVE_DIR, f"calib_{count:03d}_{ts}.png")
                cv2.imwrite(filename, frame)
                count += 1
                print(f"[拍照] 保存第 {count} 张：{filename}")
            else:
                print("[警告] 未检测到棋盘格，无法拍照")
        elif key == ord('d'):  # 删除最后一张
            files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith('.png')])
            if files:
                os.remove(os.path.join(SAVE_DIR, files[-1]))
                count -= 1
                print(f"[删除] 删除最后一张：{files[-1]}")
            else:
                print("[提示] 没有可删除的图片")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print(f"拍摄完成！共 {count} 张图片保存在 {SAVE_DIR}/ 目录")
    if count < 15:
        print("[提示] 建议至少拍 15 张以上，请继续拍摄")
    else:
        print("[信息] 图片数量充足，可以进入第3步：标定")


if __name__ == "__main__":
    main()
