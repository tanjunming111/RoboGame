# 相机标定与位姿估计 —— 使用指南

## 概述

本目录使用张正友标定法完成相机内参标定，并用 PnP/ArUco 算法实时估计相机位姿。

## 文件说明

| 文件 | 说明 |
|---|---|
| step01_generate_checkerboard.py | 生成棋盘格标定板图片 |
| step02_capture_calibration.py | 拍摄标定图片（不同角度） |
| step03_zhang_calibration.py | 张正友标定法求内参 K 和畸变 D |
| step04_verify_calibration.py | 验证标定效果（畸变校正对比） |
| step05_pnp_pose_estimation.py | 用标定结果做 PnP 实时位姿估计 |
| step06_generate_aruco.py | 生成 ArUco 标记二维码 |
| combine_aruco_print.py | 4个ArUco标记拼到A4纸方便打印裁剪 |
| step07_place_aruco_markers.py | 单标记（ID=0）场地坐标系配置 |
| step08_aruco_pose_estimation.py | 单标记（ID=0）实时位姿估计 |

## 使用步骤

### 第一阶段：相机标定（一次性）

#### 1. 生成棋盘格
```
python step01_generate_checkerboard.py
```
生成 checkerboard.png，打印到 A4 纸，测量实际方格边长（毫米）。

#### 2. 拍摄标定图片
```
python step02_capture_calibration.py
```
对着棋盘格从不同角度拍 15~30 张照片，保存在 calibration_images/ 目录。

#### 3. 张正友标定
```
python step03_zhang_calibration.py
```
自动检测角点 → calibrateCamera 求内参 → 输出标定结果到 camera_params.json。

**注意**：修改 step03 中 `SQUARE_SIZE` 为实际测量的方格边长。

#### 4. 验证标定效果
```
python step04_verify_calibration.py
```
对比原始画面和畸变校正后画面，直线应变直。

### 第二阶段：位姿估计

#### 5. 棋盘格 PnP 位姿估计（验证用）
```
python step05_pnp_pose_estimation.py
```
拿着棋盘格对着相机，实时显示位置（mm）和欧拉角（度），画面上画3D坐标轴。

### 第三阶段：ArUco 单标记位姿测试

#### 6. 生成并打印 ArUco 标记
```
python step06_generate_aruco.py
```
生成5个ArUco标记（ID=0~4），保存在 aruco_markers/ 目录。

或者用A4拼接版（推荐）：
```
python combine_aruco_print.py
```
生成 aruco_markers/aruco_4in1_A4.png，4个标记（ID=0~3）排在一张A4纸上，
每个标记周围有白边和虚线裁剪参考线。打印时选「实际大小 / 100%」不缩放。

**裁剪后用尺子测量标记黑色正方形边长**，记录实际毫米数。

#### 7. 配置标记尺寸和位置
```
python step07_place_aruco_markers.py
```

**使用方法：**

1. **修改标记尺寸**：打开 `step07_place_aruco_markers.py`，修改 `MARKER_SIZE_MM` 为你实测的边长（例如实测5.6cm则改为 `56.0`）
2. 运行脚本，自动生成 `marker_positions.json` 配置文件和 `field_layout.png` 布局图
3. 当前配置：单标记模式，只使用 ID=0，标记中心在原点 (0, 0, 0)

**坐标系定义：**
- 原点：标记中心
- X轴：向右（标记水平方向）
- Y轴：向下（标记垂直方向）
- Z轴：向前（标记法线方向，朝向相机）

#### 8. 实时单标记位姿估计
```
python step08_aruco_pose_estimation.py
```

**使用方法：**

1. 将裁剪好的 ID=0 标记平贴在桌面或地面上（不能翘起）
2. 运行脚本，打开摄像头
3. 将摄像头对准标记，距离约 20~80cm
4. 画面上会显示：
   - 绿色检测框 + ID号（确认检测到标记）
   - 标记上的3D坐标轴（红=X, 绿=Y, 蓝=Z，验证位姿方向是否正确）
   - 右上角实时显示：相机位置(X,Y,Z mm)、距离(mm)、欧拉角(P,Y,R deg)、重投影误差(px)

**画面信息含义：**

| 显示内容 | 含义 |
|---|---|
| Pos: X/Y/Z | 相机在标记坐标系下的位置（mm） |
| Dist | 相机到标记的直线距离（mm） |
| Rot: P/Y/R | 相机的俯仰/偏航/滚转角（度） |
| Reproj err | 重投影误差（px），越小越好，正常应<2px |

**验证位姿是否正确的方法：**
- 标记水平放置，相机正对往下看 → Z轴应朝上（蓝色线朝相机）
- 相机向右移动 → X值应变大
- 相机靠近标记 → Dist值应变小
- 相机绕标记旋转 → 欧拉角应相应变化

**按键：** `[q]`退出  `[s]`截图

## 输出文件

| 文件 | 内容 |
|---|---|
| camera_params.json | 内参矩阵 K, 畸变系数 D, 焦距, 主点等 |
| marker_positions.json | ArUco标记在场地中的3D坐标配置 |
| calibration_images/ | 标定图片 |
| aruco_markers/ | ArUco标记图片（含A4拼接图） |
| field_layout.png | 标记摆放示意图 |

## 标定质量判断

| 重投影误差 RMS | 质量 |
|---|---|
| < 0.5 px | 优秀 |
| 0.5~1.0 px | 良好 |
| 1.0~1.5 px | 可用 |
| > 1.5 px | 需重新标定 |
