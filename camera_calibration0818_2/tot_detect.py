# tot_detect.py
# 汇总调用 step09~step14 六个模块，一次性输出相机相对6个ArUco标记的观察情况
# ============================================================
# 功能：
#   对 ArUco.png 中的6个标记（ID=1~6）逐个调用对应模块的 get_camera_pose：
#     - 相机相对该标记的位置 position_mm = [x,y,z]（该标记坐标系，mm）
#     - 相机相对该标记的朝向 euler_deg = (pitch,yaw,roll)（度）
#     - is_detected=False 表示未观察到该标记，此时其余字段均为None
#
# 外部调用方法：
#   from tot_detect import detect_all
#
#   results = detect_all(frame)   # frame: BGR图像(np.ndarray)
#   # 或不传frame，自动打开摄像头拍一帧：
#   results = detect_all()
#
#   for mid in sorted(results):
#       r = results[mid]
#       if r['is_detected']:
#           print(mid, r['position_mm'], r['euler_deg'])  # 位置+朝向
#       else:
#           print(mid, '未检测到')
# ============================================================

import cv2
import numpy as np
import time
import math

# 六个标记模块的位姿接口（ID=1~6 对应 step09~step14）
from step09_aruco_id1_pose_estimation import (
    get_camera_pose as _pose_id1, load_camera_params)
from step10_aruco_id2_pose_estimation import get_camera_pose as _pose_id2
from step11_aruco_id3_pose_estimation import get_camera_pose as _pose_id3
from step12_aruco_id4_pose_estimation import get_camera_pose as _pose_id4
from step13_aruco_id5_pose_estimation import get_camera_pose as _pose_id5
from step14_aruco_id6_pose_estimation import get_camera_pose as _pose_id6

# 能否放下爪子的接口
from box_detector import pd_down

# 标记ID → 对应模块的位姿接口
_POSE_FUNCS = {
    1: _pose_id1,
    2: _pose_id2,
    3: _pose_id3,
    4: _pose_id4,
    5: _pose_id5,
    6: _pose_id6,
}

class state:
    def __init__(self):
        self.step = 0
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.w = 0 # 角速度，顺时针为正方向
        self.tn_ag = 0 # 总共转过的角度
        self.nd_dir = 0 # 理论上正确的方向
        self.bg = False
        self.rot = False
        self.rot_bg = 0

        self.ndx = 0
        self.ndy = 0
        self.catch_ps = 0 # 在二维码左侧还是右侧

        self.slp_time = 0 # 等待时间（等到什么时候）
        self.nxt = 0 # 下一个内容
        self.dn_high = 0 # 吸盘高度与方块高度的差（即相对高度）
        self.dn_v = 0 # 下放速度
        self.nd_catch = False
        self.nd_throw = False

wh = state()

def getspeed():
    # 返回机器人的前进速度，水平速度（向右为正方向），转弯角速度（顺时针为正方向），下放速度（负数则为上升速度），吸盘是否抓取，吸盘是否放下
    return wh.vy, wh.vx, wh.w + 0.1, wh.dn_v, wh.nd_catch, wh.nd_throw # 第三项加的是角速度修正量，需要自行调节

def giv_high(high1):
    wh.dn_high = high1

def _empty_result():
    # 未检测到时统一的结果结构（与各模块get_camera_pose未检出时一致）
    return {'is_detected': False, 'position_mm': None, 'euler_deg': None,
            'distance_mm': None, 'reproj_error_px': None,
            'rvec': None, 'tvec': None, 'corners': None,
            'detected_on': None}


def detect_all(frame=None, camera=None, marker_size_mm=None):
    # """
    # 对6个标记逐个调用step09~step14，返回相机相对每个标记的位置和朝向

    # 参数：
    #     frame  : BGR图像(np.ndarray)。为None时从camera读一帧
    #              （camera也为None时临时打开摄像头0，用完立即释放；
    #               只取一帧、六个标记共用，不会重复打开摄像头）
    #     camera : 已打开的cv2.VideoCapture对象
    #     marker_size_mm : 标记边长(mm)，为None时各模块用自身默认MARKER_SIZE_MM

    # 返回 dict：{标记ID(1~6): 该标记的result字典}，每个result字段含义：
    #     is_detected     : bool，是否观察到该标记
    #     position_mm     : np.array([x,y,z])，相机在该标记坐标系下的位置(mm)
    #     euler_deg       : (pitch,yaw,roll)，相机相对该标记的朝向（度）
    #     distance_mm     : float，相机到该标记的直线距离(mm)
    #     reproj_error_px : float，重投影误差(px)
    #     rvec / tvec     : solvePnP原始结果（标记→相机变换）
    #     corners         : 标记4个角点2D坐标(4,2)
    #     detected_on     : 'undistorted' 或 'raw'（检测路径，见各模块说明）
    #     未检测到时：is_detected=False，其余字段均为None
    # """
    # ===== 0. 获取图像帧（只取一帧，六个模块共用） =====
    if frame is None:
        temp_camera = None
        if camera is None:
            temp_camera = cv2.VideoCapture(0)  # 未传入图像源→临时打开摄像头
            camera = temp_camera
        ret, frame = camera.read()
        if temp_camera is not None:
            temp_camera.release()  # 临时打开的摄像头用完立即释放
        if not ret:
            print("[警告] 无法获取图像帧")
            return {mid: _empty_result() for mid in _POSE_FUNCS}

    # ===== 1. 读取标定参数（只读一次，传给六个模块，避免重复读文件） =====
    K, D = load_camera_params()
    if K is None:
        return {mid: _empty_result() for mid in _POSE_FUNCS}

    # ===== 2. 逐个标记调用对应模块（传K,D避免各模块再读文件） =====
    results = {}
    for mid, func in _POSE_FUNCS.items():
        results[mid] = func(frame=frame, K=K, D=D,
                            marker_size_mm=marker_size_mm)
    return results

# """在画面上画单个标记的检测框、ID和3D坐标轴（红X绿Y蓝Z）"""
def _draw_result(display, mid, r, K, marker_size_mm):
    pts = r['corners'].astype(int)
    for k in range(4):  # 检测框
        cv2.line(display, tuple(pts[k]), tuple(pts[(k + 1) % 4]),
                 (0, 255, 0), 2)
    cv2.putText(display, f"ID={mid}", (int(pts[0][0]), int(pts[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 3D坐标轴（与solvePnP一致按无畸变投影）
    axis_length = marker_size_mm * 0.8
    axis_3d = np.array([
        [0, 0, 0], [axis_length, 0, 0],
        [0, axis_length, 0], [0, 0, -axis_length],
    ], dtype=np.float64)
    axis_2d, _ = cv2.projectPoints(axis_3d, r['rvec'], r['tvec'], K, None)
    axis_2d = axis_2d.reshape(-1, 2).astype(int)
    origin = tuple(axis_2d[0])
    cv2.line(display, origin, tuple(axis_2d[1]), (0, 0, 255), 3)  # X=红
    cv2.line(display, origin, tuple(axis_2d[2]), (0, 255, 0), 3)  # Y=绿
    cv2.line(display, origin, tuple(axis_2d[3]), (255, 0, 0), 3)  # Z=蓝


def _summary_lines(results):
    # """生成左上角汇总文字：每个ID的位置(mm)和朝向(度)，未检测到则标注"""
    n = sum(1 for r in results.values() if r['is_detected'])
    lines = [f"detected: {n}/6", "ID  Pos(x,y,z mm) | P/Y/R(deg)"]
    for mid in sorted(results):
        r = results[mid]
        if r['is_detected']:
            p, a = r['position_mm'], r['euler_deg']
            lines.append(f"{mid}: {p[0]:7.1f} {p[1]:6.1f} {p[2]:7.1f} | "
                         f"{a[0]:6.1f} {a[1]:5.1f} {a[2]:6.1f}")
        else:
            lines.append(f"{mid}: --- not detected ---")
    return lines

def to_area(tmp):
    while tmp < 0:
        tmp += 2 * math.pi
    while tmp >= 2 * math.pi:
        tmp -= 2 * math.pi
        return tmp

def gt_adjust_w(tw, tdir, rs):
    cnt = 0
    tmp = 0
    pi = math.pi
    if rs[1]['is_detected']:
        cnt += 1
        tmp += to_area(rs[1]['euler_deg'][1] - pi / 2)
    if rs[2]['is_detected']:
        cnt += 1
        tmp += to_area(rs[2]['euler_deg'][1])
    if rs[3]['is_detected']:
        cnt += 1
        tmp += to_area(rs[3]['euler_deg'][1] - pi / 2)
    if rs[4]['is_detected']:
        cnt += 1
        tmp += to_area(rs[4]['euler_deg'][1])
    if rs[5]['is_detected']:
        cnt += 1
        tmp += to_area(rs[5]['euler_deg'][1] + pi / 2)
    if rs[6]['is_detected']:
        cnt += 1
        tmp += to_area(rs[6]['euler_deg'][1] + pi)

    if cnt > 0: # 用二维码更新方位
        tw = tmp / cnt
        wh.w = tw

    if tw - 0.01 <= tdir <= tw + 0.01:
        return 0
    elif tw < tdir:
        return 1
    elif tw > tdir:
        return -1

def main():
    mx_high = 100 # 这里需要实测修改

    # ===== 读取标定参数 =====
    K, D = load_camera_params()
    if K is None:
        return

    # print("tot_detect：六标记汇总观察（ArUco.png, ID=1~6, AprilTag36h11）")
    # print("调用step09~step14六个模块，每个标记独立输出：")
    # print("  相机相对该标记的位置 position_mm (mm)")
    # print("  相机相对该标记的朝向 euler_deg (P/Y/R, deg)")
    # print("  未观察到的标记 is_detected=False，其余字段为None")
    # print("画面：绿色框=检测到，框上画3D坐标轴（红X绿Y蓝Z）")
    # print("左上角为六个标记的实时汇总表，控制台每秒打印一次")
    # print("外部调用接口：detect_all(frame) → {ID: result字典}")
    # print("按键：[q]退出  [s]截图")

    # ===== 打开摄像头 =====
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    lst_time = time.time()
    while True:
        now = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # ===== 汇总检测（传原始帧，接口内部处理畸变） =====
        results = detect_all(frame=frame) # 得到所有数据

        # 按序放置
        rs = [None]*7
        for id , r in sorted(results.items()):
            rs[id] = r

        # 底图：正常情况（校正后画面检测）用校正图；
        # 全部走raw兜底路径时（如对静态图片）用原始画面。
        # 注：混合路径的少数情况，个别框可能与底图有轻微错位，
        #     数值输出（位置/朝向）不受影响
        frame_undist = cv2.undistort(frame, K, D)
        if any(r['detected_on'] == 'undistorted' for r in results.values()):
            display = frame_undist.copy()
        else:
            display = frame.copy()

        # ===== 画每个检测到的标记 =====
        for mid, r in sorted(results.items()):
            if r['is_detected']:
                _draw_result(display, mid, r, K, 56.0)

        # 更新位置和朝向
        wh.x += (wh.vx * 1000 * math.cos(wh.tn_ag) + wh.vy * 1000 * math.sin(wh.tn_ag)) * (now - lst_time)
        wh.y += (wh.vx * 1000 * math.sin(wh.tn_ag) + wh.vy * 1000 * math.cos(wh.tn_ag)) * (now - lst_time)
        wh.tn_ag += wh.w * (now - lst_time)

        if wh.rot == False: # 方向微调
            wh.w = gt_adjust_w(wh.w, wh.nd_dir * math.pi / 2, rs) * 0.1

        if wh.step == 0:
            if wh.bg == False and rs[1]['is_detected']:
                wh.y = - rs[1]['position_mm'][0]
                wh.x = - rs[1]['position_mm'][2]
                wh.bg = True
                wh.vy = 0.1
            if rs[1]['is_detected']:
                wh.y = - rs[1]['position_mm'][0]
                wh.x = - rs[1]['position_mm'][2]
            if wh.bg and wh.y >= 0:
                wh.vy = 0
                wh.step = 1 # 开始转弯
                wh.rot == True
                wh.rot_bg = now
                wh.w = math.pi / 2 / 2
                wh.nd_dir = 1
        elif wh.step == 1:
            if now - wh.rot_bg >= 1 or (rs[5]['is_detected'] and rs[5]['euler_deg'][1] >= -1):
                wh.rot = False
                wh.w = 0
                wh.step = 2
                wh.vy = 0.1
                wh.ndy = 3100
        elif wh.step == 2:
            if rs[6]['is_detected']:
                wh.ndy = wh.y - rs[6]['position_mm'][0]
            if wh.y >= wh.ndy or (rs[5]['is_detected'] and rs[5]['position_mm'][2] <= 900):
                wh.vy = 0
                wh.step = 3
                wh.rot == True
                wh.rot_bg = now
                wh.w = - math.pi / 2 / 2
                wh.nd_dir = 0
        elif wh.step == 3:
            if now - wh.rot_bg >= 1 or (rs[4]['is_detected'] and rs[4]['euler_deg'][1] <= 1):
                wh.rot = False
                wh.w = 0
                wh.step = 4
                wh.vy = 0.1
        elif wh.step == 4:
            if rs[4]['is_detected'] and - rs[4]['position_mm'][2] <= 150:# 距离可能需要修改
                wh.vy = 0
                wh.step = 5
                wh.vx = -0.1
        elif wh.step == 5:
            if pd_down(frame, "orange"): # 是否检测到摄像头中间偏上部分有方块
                wh.vx = 0
                wh.step = -1 # 抓取
                wh.nxt = 7
                wh.catch_ps = -1 # 在二维码左侧
                wh.dn_v = 0.05
            elif wh.x <= 2350:
                wh.vx = 0.1
                wh.step = 6
        elif wh.step == 6:
            if pd_down(frame, "orange"):
                wh.vx = 0
                wh.step = -1 # 抓取
                wh.nxt = 7
                wh.catch_ps = 1 # 在二维码右侧
                wh.dn_v = 0.05
            elif wh.x >= 3850:
                wh.vx = -0.1
                wh.step = 7
        elif wh.step == -1:
            if wh.dn_high == 0:
                wh.dn_v = 0
                wh.nd_catch = True
                wh.step = -2
                wh.slp_time = now + 1
            # wh.step = 7
        elif wh.step == -2:
            if now >= wh.slp_time:
                wh.nd_catch = False
                wh.step = -3
                wh.dn_v = -0.05
        elif wh.step == -3:
            if wh.dn_high >= mx_high:
                wh.dn_v = 0
                wh.step = wh.nxt
        elif wh.step == 7: # 返回至 ArUco 4 的位置
            wh.vx = - wh.catch_ps * 0.1
            if rs[4]['is_detected'] and rs[4]['position_mm'][0] >= 0:
                wh.vx = 0
                wh.step = 10
                wh.rot = True
                wh.rot_bg = now
                wh.w = - math.pi / 2 / 2
                wh.nd_dir = 2
        elif wh.step == 10:
            if now - wh.rot_bg >= 2 or (rs[6]['is_detected'] and rs[6]['euler_deg'][1] <= 1):
                wh.rot = False
                wh.w = 0
                wh.step = 11
                wh.vy = 0.1
        elif wh.step == 11:
            if rs[6]['is_detected'] and - rs[6]['position_mm'][2] <= 150: # 距离可能需要修改
                wh.vy = 0
                wh.step = -11 # 物品放下
                wh.dn_v = 0.05
                wh.nxt = 20 # 返回出发点（后面可能还要改）
        elif wh.step == -11:
            if wh.dn_high == 0:
                wh.dn_v = 0
                wh.nd_throw = True
                wh.step = -12
                wh.slp_time = now + 7 # 等待七秒放下
        elif wh.step == -12:
            if now >= wh.slp_time:
                wh.nd_throw = False
                wh.step = -13
                wh.dn_v = -0.05
        elif wh.step == -13:
            if wh.dn_high >= mx_high:
                wh.dn_v = 0
                wh.step = wh.nxt
        elif wh.step == 12:
            # wh.rot = True
            # wh.rot_bg = now
            # wh.w = -math.pi / 2 / 2
            if now - wh.rot_bg >= 1 or (rs[4]['is_detected'] and rs[4]['euler_deg'][1] <= 1): # 回头面对二维码4
                wh.rot = False
                wh.w = 0
                wh.step = 13
        elif wh.step == 20:
            wh.vx = 0.1

                
        cv2.imshow("tot_detect: 6 ArUco Markers", display)
        print(wh.step, wh.x, wh.y, wh.vx, wh.vy, wh.w)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"tot_detect_{ts}.png", display)
            print(f"[截图] 已保存 (时间戳: {ts})")

        lst_time = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
