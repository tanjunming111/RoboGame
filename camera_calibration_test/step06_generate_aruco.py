# step06_generate_aruco.py
# 标定第6步：生成 ArUco 标记二维码
# ============================================================
# ArUco 标记原理：
#   - 每个标记是一个正方形，内部黑白方格编码唯一ID
#   - DICT_4X4_50 = 4×4 内部方格，共50个不同ID（0~49）
#   - 检测时返回4个角点（左上→右上→右下→左下）+ ID号
#   - 已知标记实际尺寸 → solvePnP → 相机相对标记的位姿
#
# 场地坐标系设计：
#   在场地不同位置贴不同ID的ArUco标记
#   每个标记的3D坐标已知 → 相机看到任意标记就能定位
#   多个标记同时检测 → 加权平均提高精度
# ============================================================

import cv2
import numpy as np
import os

# ===== ArUco 参数 =====
ARUCO_DICT = cv2.aruco.DICT_4X4_50   # 字典：4×4编码，50个标记
MARKER_IDS = [0, 1, 2, 3, 4]         # 生成5个标记，ID=0~4
MARKER_PIXELS = 200                    # 生成图片的每边像素数
MARKER_SIZE_MM = 50.0                  # 打印后的实际边长（mm），测量后修改！

# 保存目录
OUTPUT_DIR = "aruco_markers"


def main():
    print("=" * 60)
    print("标定第6步：生成 ArUco 标记二维码")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取ArUco字典
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    print(f"\n字典类型：DICT_4X4_50（4×4编码，50个标记）")
    print(f"生成标记：ID = {MARKER_IDS}")
    print(f"图片尺寸：{MARKER_PIXELS}×{MARKER_PIXELS} 像素")
    print(f"标记实际边长：{MARKER_SIZE_MM} mm（打印后用尺子测量！）")
    print(f"保存目录：{OUTPUT_DIR}/\n")
    print("生成中...")

    for marker_id in MARKER_IDS:
        # 生成标记图片
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, MARKER_PIXELS
        )

        # 保存
        filename = os.path.join(OUTPUT_DIR, f"aruco_{marker_id}.png")
        cv2.imwrite(filename, marker_img)
        print(f"  [ID={marker_id}] 已保存 {filename}")

    # 生成一张包含所有标记的拼接图（方便一次性打印）
    markers_per_row = 3
    rows = (len(MARKER_IDS) + markers_per_row - 1) // markers_per_row
    gap = 20  # 标记间间距像素
    canvas_w = markers_per_row * MARKER_PIXELS + (markers_per_row + 1) * gap
    canvas_h = rows * MARKER_PIXELS + (rows + 1) * gap
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    for i, marker_id in enumerate(MARKER_IDS):
        row = i // markers_per_row
        col = i % markers_per_row
        y = gap + row * (MARKER_PIXELS + gap)
        x = gap + col * (MARKER_PIXELS + gap)

        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, MARKER_PIXELS
        )
        canvas[y:y+MARKER_PIXELS, x:x+MARKER_PIXELS] = marker_img

        # 标注ID号
        cv2.putText(canvas, f"ID={marker_id}",
                    (x + 5, y + MARKER_PIXELS + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    # 放大2倍方便打印
    canvas_large = cv2.resize(canvas, (canvas_w * 2, canvas_h * 2),
                               interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "aruco_all.png"), canvas_large)
    print(f"\n  [全部] 已保存 aruco_all.png（拼接图，放大2倍）")

    print("\n" + "=" * 60)
    print("下一步操作：")
    print("  1. 打印 aruco_all.png 或单个 aruco_X.png")
    print("  2. 打印时选「实际大小」，不要缩放")
    print("  3. 用尺子测量标记实际边长（毫米）")
    print(f"  4. 修改 step08 中的 MARKER_SIZE_MM = 实际测量值")
    print("  5. 按场地坐标系图示贴标记（见 step07 说明）")
    print("=" * 60)

    # 显示预览
    cv2.imshow("ArUco Markers", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
