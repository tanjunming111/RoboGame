box_detector_test 是正方体识别文件夹，负责识别正方体位置。

camera_calibration_test 是相机位姿判定器，负责判断相机位姿。其中的 README.md 内有使用介绍。

由于这个项目中一些部分我使用了 AI 完成，所以一些表述有 AI 味。能够使用即可。

camera_calibration 是最终需导入树莓派的内容，其中tot_detect.py为主程序（可能需要改名为 main.py），程序中有 getspeed() 函数来获取速度和转弯角速度。不过注意要将camera_calibration_test 中标定计算出的camera_params.json放入才可正常使用。

当前是粗略版，过程为从起点移动到橙色方块区域再移动到放置方块的区域，且速度一般设为 0.1m/s。后续将完善内容，加入速度判断逻辑。
