# step07_place_aruco_markers.py
# 标定第7步：场地坐标系设计 —— 单标记（ID=0）摆放说明
# ============================================================
# 原理：
#   在场地中已知位置贴一个ArUco标记（ID=0），建立场地坐标系
#   相机检测到该标记后，通过solvePnP求出相机相对标记的位姿
#   单标记测试：验证检测和位姿估计流程是否正确
#
# 场地坐标系设计（以标记中心为原点）：
#   原点：标记中心
#   X轴：向右（标记水平方向）
#   Y轴：向下（标记垂直方向，朝向相机方向）
#   Z轴：向前（标记法线方向，朝向相机）
#
# 本文件：可视化场地布局 + 配置标记位置
# ============================================================

import cv2
import numpy as np
import json
import os

# ===== ArUco 标记位置配置 =====
# 只用 ID=0 一个标记
# 标记放在场地原点，平贴在地面/桌面上
# pos = [x, y, z] 坐标（mm），z=0 表示贴在地面上
MARKER_POSITIONS = {
    0: {'pos': [0, 0, 0], 'desc': '原点标记'},
}

MARKER_SIZE_MM = 56.0  # 标记实际边长（mm），实测5.6cm

# 输出配置文件
OUTPUT_FILE = "marker_positions.json"


def draw_field_layout():
    """绘制标记摆放示意图"""
    img_w, img_h = 500, 400
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240

    # 画坐标系
    cx, cy = 250, 200  # 图像中心 = 标记位置
    cv2.arrowedLine(img, (cx, cy), (cx + 80, cy), (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(img, "X", (cx + 85, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.arrowedLine(img, (cx, cy), (cx, cy - 80), (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(img, "Y", (cx - 15, cy - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.arrowedLine(img, (cx, cy), (cx - 50, cy + 50), (255, 0, 0), 2, tipLength=0.3)
    cv2.putText(img, "Z", (cx - 65, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 画标记方块
    ms = 80  # 标记在图上的大小
    cv2.rectangle(img, (cx - ms//2, cy - ms//2), (cx + ms//2, cy + ms//2),
                   (0, 100, 255), -1)
    cv2.rectangle(img, (cx - ms//2, cy - ms//2), (cx + ms//2, cy + ms//2),
                   (0, 0, 0), 2)

    # 标注ID和坐标
    cv2.putText(img, "ID=0", (cx - 20, cy - ms//2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "(0, 0, 0) mm", (cx - 40, cy + ms//2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # 标题
    cv2.putText(img, "Single Marker Layout (ID=0)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, f"Marker size: {MARKER_SIZE_MM}mm",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(img, "Orange = ArUco marker at origin",
                (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    return img


def main():
    print("=" * 60)
    print("标定第7步：单标记场地坐标系设计（ID=0）")
    print("=" * 60)

    print(f"\n标记数量：1个（ID=0）")
    print(f"标记边长：{MARKER_SIZE_MM}mm")
    print(f"\n标记位置配置：")
    print("-" * 50)
    for mid, info in MARKER_POSITIONS.items():
        x, y, z = info['pos']
        print(f"  ID={mid}: ({x:5.0f}, {y:5.0f}, {z:5.0f}) mm  - {info['desc']}")
    print("-" * 50)

    # 保存配置到JSON
    config = {
        'field_width_mm': 0,  # 单标记模式不使用场地尺寸
        'field_height_mm': 0,
        'marker_size_mm': MARKER_SIZE_MM,
        'markers': {}
    }
    for mid, info in MARKER_POSITIONS.items():
        config['markers'][str(mid)] = {
            'position_mm': info['pos'],
            'description': info['desc']
        }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 配置已保存到 {OUTPUT_FILE}")

    # 绘制场地布局图
    layout = draw_field_layout()
    layout_path = "field_layout.png"
    cv2.imwrite(layout_path, layout)
    print(f"[保存] 场地图已保存到 {layout_path}")

    print("\n" + "=" * 60)
    print("摆放步骤：")
    print("  1. 打印ArUco标记（combine_aruco_print.py生成的A4图）")
    print("  2. 裁下ID=0的标记，平贴在桌面/地面上")
    print("  3. 标记必须平贴，不能翘起")
    print("  4. 标记中心即为场地原点 (0, 0, 0)")
    print("  5. 运行step08检测标记并估计相机位姿")
    print("=" * 60)

    # GUI预览
    try:
        cv2.imshow("Field Layout", layout)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("[提示] 当前环境不支持GUI显示，场地图已保存到 field_layout.png")


if __name__ == "__main__":
    main()
