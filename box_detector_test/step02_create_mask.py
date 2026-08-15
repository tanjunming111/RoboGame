# step02_create_mask.py
# 第二步：用 Trackbar 动态调节 HSV 阈值，生成掩码图
# 说明：通过滑动条实时调节 HSV 上下界，观察掩码变化
# 目的：找到准确的橙色 HSV 阈值范围
# 用法：python step02_create_mask.py，拖动滑动条调阈值，按 q 退出

import cv2
import numpy as np


def nothing(x):
    """Trackbar 回调函数，什么都不做"""
    pass


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 创建窗口
    cv2.namedWindow("Step 02: Create Mask")
    cv2.namedWindow("Trackbars")

    # 创建滑动条
    # H(色相)范围：0~179
    cv2.createTrackbar("H_Low",  "Trackbars", 10, 179, nothing)
    cv2.createTrackbar("H_High", "Trackbars", 25, 179, nothing)
    # S(饱和度)范围：0~255
    cv2.createTrackbar("S_Low",  "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("S_High", "Trackbars", 255, 255, nothing)
    # V(亮度)范围：0~255
    cv2.createTrackbar("V_Low",  "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("V_High", "Trackbars", 255, 255, nothing)

    print("=" * 50)
    print("第二步：生成掩码图")
    print("=" * 50)
    print("操作说明：")
    print("  1. 把橙色方块放在摄像头前")
    print("  2. 拖动 H_Low/H_High：只让橙色区域变白")
    print("  3. 拖动 S_Low/V_Low：如果白色区域有洞，适当降低")
    print("目标：掩码图(Mask)中，方块是纯白色，背景是纯黑色")
    print("按 [q] 退出")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 读取滑动条的值
        h_low  = cv2.getTrackbarPos("H_Low",  "Trackbars")
        h_high = cv2.getTrackbarPos("H_High", "Trackbars")
        s_low  = cv2.getTrackbarPos("S_Low",  "Trackbars")
        s_high = cv2.getTrackbarPos("S_High", "Trackbars")
        v_low  = cv2.getTrackbarPos("V_Low",  "Trackbars")
        v_high = cv2.getTrackbarPos("V_High", "Trackbars")

        # BGR → HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 构建阈值范围
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])

        # 生成掩码：在范围内的像素变为白色(255)，其余变为黑色(0)
        mask = cv2.inRange(hsv, lower, upper)

        # 把掩码转成彩色，方便并排显示
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 在画面上显示当前阈值参数
        info = f"H:[{h_low},{h_high}] S:[{s_low},{s_high}] V:[{v_low},{v_high}]"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Original", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(mask_color, "Mask (white=detected)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 并排显示
        combined = np.hstack([frame, mask_color])
        cv2.imshow("Step 02: Create Mask", combined)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 退出时打印最终参数，供你复制到下一步使用
    h_low  = cv2.getTrackbarPos("H_Low",  "Trackbars")
    h_high = cv2.getTrackbarPos("H_High", "Trackbars")
    s_low  = cv2.getTrackbarPos("S_Low",  "Trackbars")
    s_high = cv2.getTrackbarPos("S_High", "Trackbars")
    v_low  = cv2.getTrackbarPos("V_Low",  "Trackbars")
    v_high = cv2.getTrackbarPos("V_High", "Trackbars")

    print("\n" + "=" * 50)
    print("你调好的阈值参数：")
    print(f"  LOWER = np.array([{h_low}, {s_low}, {v_low}])")
    print(f"  UPPER = np.array([{h_high}, {s_high}, {v_high}])")
    print("=" * 50)


if __name__ == "__main__":
    main()
