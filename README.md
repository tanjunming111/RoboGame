注意所有 camera_calibration 的修改要在最新版（每个camera_calibration文件夹有标注日期和版本号）中修改！

# 待做事项
注意在开始前，需要完成以下准备工作。

## 一、测量颜色的 HSV 阈值
进入box_detector_test 文件夹中， 打开 step02_create_mask.py，测量橙色和紫色方块在赛场上颜色的 HSV 具体步骤如下：
- 第一步：
在step02_create_mask.py中调节HSV值。一开始把范围调到最大（min=0，max=255），然后按H,S,V的顺序将两边收紧，直到阈值只包含方块的颜色。
这个阈值是为了在有光照和阴影影响的时候也能识别方块，同一方块在不同的背景和环境中的阈值会有差别，建议在比赛的台上测，以模拟实际情况。
- 第二步：
将这个阈值放到 box.py 中，用摄像头即可识别方块轮廓。

## 二、张正友标定法标定
在 camera_calibration_test文件夹中，按 README.md 中的内容做，得到camera_params.json。如果来不及，可以直接用camera_calibration中的camera_params.json。
一些注意事项：
- 1.注意修改 camera_calibration 文件中 step07_place_aruco_markers.py 第31行的二维码边长参数，单位是 mm。
- 2.需要得到的文件为 camera_params.json。

## 三、camera_calibration 内容更新
更新：
- 1.box_detector_orange.py、box_detector_purple.py 中的颜色HSV阈值
- 2.step09_aruco_id1_pose_estimation.py 到 step14_aruco_id1_pose_estimation.py 六个文件中第 40 行的二维码边长
- 3.camera_params.json。

## 四、camera_calibration 内容调试
- 1.box_detector.py 中需要调试 圆心位置cx,cy、圆形半径 RR 和占比 ratio >= 0.95 的数值 0.95
- 2.tot_detect.py 中需要调试角速度修正量，在 getspeed() 中修改
- 3.吸盘和方块相对高度在giv_high导入（可能需要修改格式），并确定吸盘上升高度的最大值

# 项目介绍

box_detector_test 是正方体识别文件夹，负责识别正方体位置。

camera_calibration_test 是相机位姿判定器，负责判断相机位姿。其中的 README.md 内有使用介绍。

由于这个项目中一些部分我使用了 AI 完成，所以一些表述有 AI 味。能够使用即可。

camera_calibration 是最终需导入树莓派的内容，其中tot_detect.py为主程序（可能需要改名为 main.py），程序中有 getspeed() 函数来获取速度和转弯角速度。不过注意要将camera_calibration_test 中标定计算出的camera_params.json放入才可正常使用。

当前是粗略版，过程为从起点移动到橙色方块区域再移动到放置方块的区域，且速度一般设为 0.1m/s。后续将完善内容，加入速度判断逻辑。
