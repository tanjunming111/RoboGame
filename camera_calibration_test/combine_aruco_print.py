# combine_aruco_print.py
# 将前4张ArUco标记（ID=0~3）拼到一张A4纸大小的图中
# 布局：2行×2列，每个标记周围留充足白边方便裁剪
# A4尺寸：210mm×297mm，300DPI → 2480×3508像素

import cv2
import numpy as np
import os

# ===== A4 纸参数 =====
# A4 = 210mm × 297mm，打印分辨率300DPI
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
DPI = 300  # 打印分辨率
MM_TO_PX = DPI / 25.4  # 毫米转像素的系数

A4_WIDTH_PX = int(A4_WIDTH_MM * MM_TO_PX)   # 2480
A4_HEIGHT_PX = int(A4_HEIGHT_MM * MM_TO_PX) # 3508

# ===== ArUco 参数 =====
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]

# ===== 布局参数 =====
MARKER_SIZE_MM = 60     # 标记实际打印边长（mm），打印后用尺子校准
MARKER_SIZE_PX = int(MARKER_SIZE_MM * MM_TO_PX)  # 标记像素尺寸
WHITE_MARGIN_MM = 12   # 每个标记周围的白色边距（mm），方便裁剪
WHITE_MARGIN_PX = int(WHITE_MARGIN_MM * MM_TO_PX)
GAP_MM = 10             # 标记之间的间距（mm）
GAP_PX = int(GAP_MM * MM_TO_PX)

# 保存路径
OUTPUT_DIR = "aruco_markers"
OUTPUT_FILE = "aruco_4in1_A4.png"


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=10):
    """画虚线"""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = int(np.hypot(dx, dy))
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    n_dashes = length // (dash_len * 2)
    for i in range(n_dashes + 1):
        s = i * dash_len * 2
        e = min(s + dash_len, length)
        sx, sy = int(x1 + ux * s), int(y1 + uy * s)
        ex, ey = int(x1 + ux * e), int(y1 + uy * e)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def main():
    print("=" * 60)
    print("ArUco 标记 A4 打印图（前4张合一）")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 生成ArUco字典和4个标记
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    markers = []
    for marker_id in MARKER_IDS:
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, MARKER_SIZE_PX
        )
        markers.append(marker_img)
        print(f"  [ID={marker_id}] 已生成 {MARKER_SIZE_PX}×{MARKER_SIZE_PX} px"
              f"（{MARKER_SIZE_MM}mm）")

    # ===== 计算布局 =====
    # 每个单元格 = 白边 + 标记 + 白边
    cell_px = WHITE_MARGIN_PX * 2 + MARKER_SIZE_PX
    # 网格总尺寸
    grid_w = 2 * cell_px + GAP_PX
    grid_h = 2 * cell_px + GAP_PX

    # 居中放置在A4纸上
    margin_x = (A4_WIDTH_PX - grid_w) // 2
    margin_y = (A4_HEIGHT_PX - grid_h) // 2

    print(f"\nA4 画布：{A4_WIDTH_PX}×{A4_HEIGHT_PX} px"
          f"（{A4_WIDTH_MM}×{A4_HEIGHT_MM}mm @ {DPI}DPI）")
    print(f"标记尺寸：{MARKER_SIZE_MM}mm（{MARKER_SIZE_PX}px）")
    print(f"白边宽度：{WHITE_MARGIN_MM}mm（{WHITE_MARGIN_PX}px）")
    print(f"网格总尺寸：{grid_w}×{grid_h} px")
    print(f"页面边距：左右 {margin_x}px，上下 {margin_y}px")

    # ===== 创建A4白色画布 =====
    canvas = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype=np.uint8) * 255

    # ===== 放置4个标记 =====
    for i, marker_id in enumerate(MARKER_IDS):
        row = i // 2  # 0或1
        col = i % 2   # 0或1

        # 单元格左上角坐标
        cell_x = margin_x + col * (cell_px + GAP_PX)
        cell_y = margin_y + row * (cell_px + GAP_PX)

        # 标记在单元格内居中（四周是白边）
        marker_x = cell_x + WHITE_MARGIN_PX
        marker_y = cell_y + WHITE_MARGIN_PX

        # 贴标记到画布
        canvas[marker_y:marker_y + MARKER_SIZE_PX,
               marker_x:marker_x + MARKER_SIZE_PX] = markers[i]

        # 在标记下方标注ID号（白边区域内）
        label_y = marker_y + MARKER_SIZE_PX + int(8 * MM_TO_PX)
        label_x = marker_x + MARKER_SIZE_PX // 2 - 50
        cv2.putText(canvas, f"ID={marker_id}",
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

        # 在标记上方标注尺寸信息
        info_y = marker_y - int(8 * MM_TO_PX)
        cv2.putText(canvas, f"{MARKER_SIZE_MM}mm",
                    (label_x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 100, 1)

        # 画裁剪参考线（虚线框，比单元格稍小一点）
        cut_inset = int(3 * MM_TO_PX)  # 裁剪线距单元格边缘3mm
        dashed_color = 200  # 浅灰色
        x1 = cell_x + cut_inset
        y1 = cell_y + cut_inset
        x2 = cell_x + cell_px - cut_inset
        y2 = cell_y + cell_px - cut_inset
        # 四条虚线边
        draw_dashed_line(canvas, (x1, y1), (x2, y1), dashed_color)
        draw_dashed_line(canvas, (x2, y1), (x2, y2), dashed_color)
        draw_dashed_line(canvas, (x2, y2), (x1, y2), dashed_color)
        draw_dashed_line(canvas, (x1, y2), (x1, y1), dashed_color)

    # ===== 保存 =====
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    cv2.imwrite(output_path, canvas)
    print(f"\n  已保存：{output_path}")
    print(f"  文件尺寸：{A4_WIDTH_PX}×{A4_HEIGHT_PX} 像素（A4 @ 300DPI）")
    print(f"\n打印方法：")
    print(f"  1. 打印时选「实际大小」或「100%」，不要缩放")
    print(f"  2. 纸张方向：纵向（Portrait）")
    print(f"  3. 打印后沿虚线裁剪，每个标记单独使用")
    print(f"  4. 用尺子测量标记实际边长，修改 step08 中 MARKER_SIZE_MM")
    print("=" * 60)

    # GUI预览（部分环境不支持imshow，用try保护）
    try:
        # 预览时缩小到屏幕可见大小
        preview_scale = 800 / A4_HEIGHT_PX
        preview = cv2.resize(canvas, None, fx=preview_scale, fy=preview_scale,
                             interpolation=cv2.INTER_AREA)
        cv2.imshow("ArUco A4 Print Preview", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("[提示] 当前环境不支持GUI显示，图片已保存，请直接打开文件查看。")


if __name__ == "__main__":
    main()
