# step01_generate_checkerboard.py
# 标定第1步：生成棋盘格标定板图片
# ============================================================
# 张正友标定法需要一张已知尺寸的棋盘格图案
# 原理：
#   - 棋盘格的内角点（黑白方块交界处）是已知的3D点
#   - 标定时自动检测这些角点，与3D坐标对应，求解相机内参
#   - 棋盘格要求：平面、刚体、已知方格边长（毫米）
# 用法：
#   python step01_generate_checkerboard.py
#   生成 checkerboard.png 后打印到 A4 纸上，用尺子测量实际方格边长
# ============================================================

import cv2
import numpy as np

# ===== 棋盘格参数 =====
# 内角点数：9列×6行（即10×7个方格）
# 内角点 = 黑白方块交界处的角，不是方块本身的角
CHESSBOARD_COLS = 9    # 内角点列数
CHESSBOARD_ROWS = 6    # 内角点行数

# 每个方格的像素大小（生成图片用，与实际打印大小无关）
SQUARE_PIXELS = 50     # 每个方格 50 像素

# 计算图片尺寸（加上一个方格的边框）
img_cols = (CHESSBOARD_COLS + 1) * SQUARE_PIXELS  # 10个方格宽
img_rows = (CHESSBOARD_ROWS + 1) * SQUARE_PIXELS  # 7个方格高

# ===== 生成棋盘格图片 =====
# 先创建白色背景
board = np.ones((img_rows, img_cols), dtype=np.uint8) * 255

# 交替填充黑色方格
for row in range(CHESSBOARD_ROWS + 1):
    for col in range(CHESSBOARD_COLS + 1):
        # (row + col) 为奇数时画黑色方格
        if (row + col) % 2 == 1:
            y1 = row * SQUARE_PIXELS
            y2 = (row + 1) * SQUARE_PIXELS
            x1 = col * SQUARE_PIXELS
            x2 = (col + 1) * SQUARE_PIXELS
            board[y1:y2, x1:x2] = 0  # 黑色

# 保存图片
output_path = "checkerboard.png"
# 放大到适合打印的尺寸（A4纸约 2480×3508 像素 @300dpi）
scale = 4
board_large = cv2.resize(board, (img_cols * scale, img_rows * scale),
                          interpolation=cv2.INTER_NEAREST)
cv2.imwrite(output_path, board_large)

print("=" * 60)
print("标定第1步：生成棋盘格标定板")
print("=" * 60)
print()
print(f"棋盘格参数：")
print(f"  内角点数：{CHESSBOARD_COLS}列 × {CHESSBOARD_ROWS}行 = {CHESSBOARD_COLS * CHESSBOARD_ROWS}个角点")
print(f"  方格数：{CHESSBOARD_COLS + 1} × {CHESSBOARD_ROWS + 1}")
print(f"  生成图片：{output_path}（{img_cols * scale}×{img_rows * scale}像素）")
print()
print("下一步操作：")
print("  1. 打开 checkerboard.png，打印到 A4 纸上")
print("  2. 打印时选择「实际大小」，不要缩放")
print("  3. 用尺子测量打印后方格的实际边长（毫米）")
print("  4. 记住这个值，标定时要用（如 23.5mm）")
print("  5. 将棋盘格平贴在硬纸板上，确保完全平整")
print()
print("注意事项：")
print("  - 棋盘格必须平整，不能弯曲褶皱")
print("  - 打印后必须测量实际方格大小，不能假设是某个值")
print("  - 建议用相纸或厚卡纸打印，避免纸张变形")
print("=" * 60)

# 显示预览
cv2.imshow("Checkerboard", board)
cv2.waitKey(0)
cv2.destroyAllWindows()
