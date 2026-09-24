"""

tot_detect.py - Master control program for ArUco navigation and block handling.

Uses two cameras (front and down) and communicates with STM32 via serial.

Step motors are now controlled via relative commands (send_move_relative).

Actuator is controlled in open-loop via direction in send _ command (z_dir).

"""

import cv2

import numpy as np

import time

import math

import sys


# Import pose estimation modules

from step09_aruco_id1_pose_estimation import (get_camera_pose as _pose_id1, load_camera_params)

from step10_aruco_id2_pose_estimation import get_camera_pose as _pose_id2

from step11_aruco_id3_pose_estimation import get_camera_pose as _pose_id3

from step12_aruco_id4_pose_estimation import get_camera_pose as _pose_id4

from step13_aruco_id5_pose_estimation import get_camera_pose as _pose_id5

from step14_aruco_id6_pose_estimation import get_camera_pose as _pose_id6


# Import block detection module

from box_detector import pd_down


# Import communication module

try:

    from stm32_comm import STM32Comm

except ImportError:

    class STM32Comm:

        def __init__(self, port, baudrate, enable):

            print("[WARNING] Communication module not loaded, cannot send commands.")

        def send_command(self, *args):

            pass

        def send_move_relative(self, *args):

            pass

        def close(self):

            pass


# Global base speed (m/s)

BASE_SPEED = 0.25

# Camera indices (adjust according to your system)

CAM_DOWN  = 0   # Downward camera

CAM_FRONT = 4   # Forward camera

CAM_LEFT = 2 # Left camera



# Marker ID -> pose function mapping

_POSE_FUNCS = {
    1: _pose_id1,
    2: _pose_id2,
    3: _pose_id3,
    4: _pose_id4,
    5: _pose_id5,
    6: _pose_id6,
}


class State:

    def __init__(self):

        self.platform = 0  # platform position

        self.cap0 = None  # down
        self.cap4 = None  # front
        self.cap2 = None  # left
        self.K = None
        self.D = None
        self.hold = 0 # robot is holding the box

        # above is need

        self.step = 0
        self.x = 0.0        # Global X (mm)
        self.y = 0.0        # Global Y (mm)
        self.vx = 0.0       # Lateral velocity (m/s)
        self.vy = 0.0       # Forward velocity (m/s)
        self.w = 0.0        # Angular velocity (rad/s)
        self.tn_ag = 0.0    # Total turned angle (rad)
        self.nd_dir = 0     # Theoretical direction (0:forward, 1:right, 2:back, 3:left)
        self.bg = False     # Localization done
        self.rot = False    # Rotating flag
        self.rot_bg = 0     # Rotation start time
        self.ndx = 0
        self.ndy = 0
        self.catch_ps = 0   # -1 left, 1 right
        self.slp_time = 0
        self.nxt = 0
        self.dn_high = 0    # Actuator target height (mm) - not used in open-loop
        self.dn_v = 0.0     # Actuator direction: +1 extend, -1 retract, 0 stop
        self.nd_catch = False
        self.nd_throw = False
        self.stat = True


wh = State()

def getspeed():

    return wh.vy, wh.vx, wh.w, wh.dn_v, wh.nd_catch, wh.nd_throw


def _empty_result():
    return {'is_detected': False, 'position_mm': None, 'euler_deg': None,
            'distance_mm': None, 'reproj_error_px': None,
            'rvec': None, 'tvec': None, 'corners': None, 'detected_on': None}


def _pose_id(tb, frame, K, D):
    if tb == 1:
        return _pose_id1(frame=frame, K=K, D=D)
    elif tb == 2:
        return _pose_id2(frame=frame, K=K, D=D)
    elif tb == 3:
        return _pose_id3(frame=frame, K=K, D=D)
    elif tb == 4:
        return _pose_id4(frame=frame, K=K, D=D)
    elif tb == 5:
        return _pose_id5(frame=frame, K=K, D=D)
    elif tb == 6:
        return _pose_id6(frame=frame, K=K, D=D)

def stop_it(stm32):
    stm32.send_command(0, 0, 0, 0, 0,0, wh.hold)
    time.sleep(0.02)
    stm32.send_command(0, 0, 0, 0, 0,0, wh.hold)
    time.sleep(0.02)
    stm32.send_command(0, 0, 0, 0, 0,0, wh.hold)


def g_left(tim, tb, stm32, sl = 0, cp = 1):
    K, D = load_camera_params()
    start = time.time()
    if sl == 1:
        vr = 0.5
    else:
        vr = 0.8
    cap = wh.cap4
    if cp == 2:
        cap = wh.cap2
    ts = 0
    while ts < tim:
        stm32.send_command(0, 0, 0, x_mm=0, y_mm=0, z_dir=0, fan=wh.hold)
        time.sleep(0.2)
        ret,frame = cap.read()
        if ret == True:
            cv2.imshow("show",frame)
            cv2.waitKey(1)
            result = _pose_id(tb, frame, K, D)
            print("det:",result['is_detected'])
            if result['is_detected']:
                print(result['euler_deg'][1])
                print("euler_deg", result['euler_deg'][1])
                if result['euler_deg'][1] <= 5: # Standard
                    break
        stm32.send_command(0, 0, vr, x_mm=0, y_mm=0, z_dir=0, fan=wh.hold)
        time.sleep(0.3)
        ts += 0.25

    stop_it(stm32)

def g_right(tim, tb, stm32, sl = 0, cp = 1):
    K, D = load_camera_params()
    if sl == 1:
        vr = -0.5
    else:
        vr = -0.8
    cap = wh.cap4
    if cp == 2:
        cap = wh.cap2
    ts = 0
    start = time.time()
    while ts < tim:
        stm32.send_command(0, 0, 0, x_mm=0, y_mm=0, z_dir=0, fan=wh.hold)
        time.sleep(0.2)
        ret,frame = cap.read()
        if ret == True:
            cv2.imshow("show",frame)
            cv2.waitKey(1)
            result = _pose_id(tb, frame, K, D)
            if result['is_detected']:
                print(result['euler_deg'][1])
                if result['euler_deg'][1] >= -5:  # Standard
                    print("G_RIGHT_YES",result['euler_deg'][1])
                    # print(ts)
                    # print(time.time() - start)
                    break

        stm32.send_command(0, 0, vr, x_mm=0, y_mm=0, z_dir=0, fan=wh.hold)
        time.sleep(0.3)
        ts += 0.25

    stop_it(stm32)


def sscmd(cmd,stm32, nstop = 0):

    if len(cmd) == 6:
        vx, vy, vrot, z_dir, fan, dur = cmd
        print(f"[DEBUG] Sending command: vx={vx}, vy={vy}, vrot={vrot}, z_dir={z_dir}, fan={fan}, dur={dur}")
        start = time.time()
        while time.time() - start < dur:

            stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=wh.hold)

            time.sleep(0.02)

    elif len(cmd) == 2:
        vx,vy = cmd
        stm32.send_move_relative(vx, vy) # right is +
        slp = (vx + vy) / 40 # waiting to do the moving
        time.sleep(slp) # waiting to do the moving

    # Send relative move commands (separate)
    if nstop == 0:
        stop_it(stm32)

def gt_photo(id):
    print(id)
    cap = wh.cap0
    if id == 1:
        cap = wh.cap4
    elif id == 2:
        cap = wh.cap2
    ret,frame = cap.read()
    cv2.imshow("show",frame)
    cv2.waitKey(1)

def adjust_clock(tim, tb, stm32):
    K = wh.K
    D = wh.D
    cap = wh.cap4
    ret,frame = cap.read()
    if ret == False:
        print("adjust_clock not detect")
        return
    cv2.imshow("show",frame)
    cv2.waitKey(1)
    result = _pose_id(tb, frame, K, D)
    if result['is_detected'] == False:
        print("not detect")
        return
    if result['euler_deg'][1] > 2:
        g_left(tim, tb, stm32,sl = 1)
    elif result['euler_deg'][1] < -2:
        g_right(tim, tb, stm32,sl = 1)

    stop_it(stm32)

def adjust_face(tim, tb, stm32):
    K = wh.K
    D = wh.D
    cap = wh.cap4
    ret,frame = cap.read()
    if ret == False:
        print("adjust_face not detect")
        return
    result = _pose_id(tb, frame, K, D)
    if result['is_detected'] == False:
        print("CAN'T FIND")
        return
    if result['position_mm'][0] < -1:
        to_go = 0.5
    elif result['position_mm'][0] > 1:
        to_go = -0.5
    else:
        return
    stm32.send_command(0, to_go, 0, 0, 0,0, wh.hold)
    start = time.time()
    while time.time() - start < tim:
        ret,frame = cap.read()
        if ret == True:
            cv2.imshow("show",frame)
            cv2.waitKey(1)
            result = _pose_id(tb, frame, K, D)
            if ret == False or result['is_detected'] == False:
                print("not detect")
                break
            print(result['position_mm'][0], to_go)
            if result['position_mm'][0] * to_go >= 0:
                print("Done")
                break

        time.sleep(0.02)

    stop_it(stm32)


def go_until(tim, tb, dis, stm32):
    K = wh.K
    D = wh.D
    cap = wh.cap4
    ret,frame = cap.read()
    if ret == False:
        print("not detect")
        return
    stm32.send_command(0.5, 0, 0, 0, 0,0, wh.hold)
    start = time.time()
    while time.time() - start < tim:
        ret,frame = cap.read()
        if ret == False:
            print("not detect")
            break
        cv2.imshow("show",frame)
        cv2.waitKey(1)
        result = _pose_id(tb, frame, K, D)
        if result['is_detected']:
            print(result['position_mm'][2])
            if result['position_mm'][2] * (-1) <= dis:
                break

    stop_it(stm32)

def turn_until(tim, tb, stm32):
    K = wh.K
    D = wh.D
    cap = wh.cap4
    ret,frame = cap.read()
    if ret == False:
        print("not detect")
        return
    result = _pose_id(tb, frame, K, D)
    if result['is_detected'] == False:
        stop_it(stm32)
        return
    height, width = frame.shape[:2]
    corners = result['corners']
    center_x = float(corners[:, 0].mean())
    if center_x - width / 2 < -5:
        to_go = 0.8
    elif center_x - width / 2 > 5:
        to_go = -0.8
    else:
        return
    print("TURN TO ARUCO")
    stm32.send_command(0, 0, to_go, 0, 0,0, wh.hold)
    start = time.time()
    while time.time() - start < tim:
        ret,frame = cap.read()
        if ret == False:
            print("not detect")
            break
        cv2.imshow("show",frame)
        cv2.waitKey(1)
        result = _pose_id(tb, frame, K, D)
        if result['is_detected']:
            corners = result['corners']
            center_x = float(corners[:, 0].mean())

            if  (center_x - width / 2) * to_go >= 0:
                print("DONE")
                break
        time.sleep(0.02)

    stop_it(stm32)


def go_and_get(stm32):# keep the stuff on the top
    # front,left,un-clock,up_only_f1_0_1,air_open_01,time
    sscmd((0, 30), stm32)
    cmd = (0.5,0,0,0,0,3.5)
    vx, vy, vrot, z_dir, fan, dur = cmd
    fd = False
    sscmd((0,0,0,1,0,3),stm32)
    start = time.time()
    cap_down = wh.cap0
    while time.time() - start < dur:
        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=wh.hold)
        ret,frame = cap_down.read()
        if ret == True:
            is_det = pd_down(frame,"orange")
            if is_det:
                stop_it(stm32)
                fd = True
                break
        
        time.sleep(0.02)

    if fd == False:
        return
    
    sscmd((0,0,0,-1,0,3),stm32)
    sscmd((0,0,0,0,1,3),stm32)
    sscmd((0,0,0,1,1,3),stm32)
    sscmd((0,-30),stm32)
    sscmd((0,0,0,-1,1,2),stm32)

def get_up_slope(stm32):
    cmd = (1,0,0,0,0,2.5)
    sscmd(cmd,stm32,nstop = 1)
    cmd = (0.5,0,0,0,0,0.5) # get the speed low
    sscmd(cmd,stm32)

def get_down_slope(stm32, td = 1):
    cmd = (0.4 * td,0,0,0,0,3)
    sscmd(cmd,stm32)

normal_quarter = 2
gs_quarter = 3

def turn_left(stm32):
    sscmd((0,0,1,0,0,normal_quarter),stm32)

def turn_right(stm32): # for 1/4 round
    sscmd((0,0,-1,0,0,normal_quarter),stm32)


def set_begin(stm32):
    sscmd((0,30),stm32) # to prepare
    sscmd((0,30),stm32)
    time.sleep(1.0)
    sscmd((0,40),stm32)
    time.sleep(1.0)
    sscmd((0,-20),stm32) # put it in -20


def go_from_begin(stm32):
    sscmd((0,-0.7,0,0,0,2.0), stm32) # give some space
    print("GIVING SPACE")
    K = wh.K
    D = wh.D
    cmd = (0.5,0,0,0,0,2.5)
    vx, vy, vrot, z_dir, fan, dur = cmd
    fd = False
    start = time.time()
    cap = wh.cap2
    while time.time() - start < dur:
        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=wh.hold)
        ret,frame = cap.read()
        if ret == True:
            cv2.imshow("show",frame)
            cv2.waitKey(1)
            result = _pose_id(1, frame, K, D)
            if result['is_detected']:
                print("SEE")
                if result['position_mm'][0] <= 30:
                    print("Dist to ArUco2",result['position_mm'][0])
                    break
        time.sleep(0.02)

    print("FINISH FIRST WALKING")
    stop_it(stm32)


def left_to_right(stm32, tim, dist):
    go_until(tim, 5, dist, stm32)

def right_to_left(stm32, tim):
    cmd = (0.5,0,0,0,0,tim)
    vx, vy, vrot, z_dir, fan, dur = cmd
    stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=wh.hold)
    start = time.time()
    while time.time() - start < dur:
        ret,frame = wh.cap4.read()
        if ret == False:
            time.sleep(0.02)
            continue
        cv2.imshow("show",frame)
        cv2.waitKey(1)
        result = _pose_id(1, frame, wh.K, wh.D)
        if result['is_detected']:
            if result['position_mm'][2] * (-1) <= 1000:
                break
        time.sleep(0.02)

    stop_it(stm32)

def find_and_catch(clr, stm32):
    # Move forward until the block is detected, then catch it
    ### have to guarantee that the catcher went ahead
    crs = "orange"
    ztim = 5
    dt = 0.5
    if clr == 1:
        crs = "purple"
        ztim = 3
    ups = 2.8
    dwns = 2.5
    sscmd((0, 0, 0, 1, 0, ups), stm32)
    sscmd((0.5, 0, 0, 0, 0, 1.5), stm32)
    sscmd((0.35, 0, 0, 0, 0, 1), stm32)
    sscmd((0.8,0,0,0,0,2),stm32) # for test and wait to del
    sscmd((-0.5,0,0,0,0,0.2),stm32)
    stop_it(stm32)

    cap_down = wh.cap0
    for _ in range(5):
        cap_down.read()
        time.sleep(0.1)

    ret, frame = cap_down.read()
    if ret == True:
        if pd_down(frame,crs): # See at first
            stop_it(stm32)
            sscmd((0, 0, 0, 0, 0, 0.1), stm32)   # Stop
            sscmd((0, 0, 0, -1, 0, dwns), stm32)  # Lower the actuator
            wh.hold = 1
            sscmd((0, 0, 0, 0, 1, 2), stm32)   # Activate suction
            sscmd((0, 0, 0, 1, 1, ups), stm32)   # Raise the actuator
            sscmd((-0.5,0,0,0,0,0.2),stm32) # little back
            # sscmd((0, -10), stm32)             # Move backward
            # sscmd((0, 0, 0, -1, 1, 2), stm32)   # Lower the actuator and keep suction on
            sscmd((0.8,0,0,0,0,0.5),stm32) # for more
            sscmd((-0.5,0,0,0,0,3),stm32) # go back
            sscmd((0,-20), stm32)
            return
        
    # sscmd((0, 30), stm32)             # Move ahead
    cmd = (0.05, 0.5, 0.0, 0, 0, ztim)
    vx, vy, vrot, z_dir, fan, dur = cmd
    fd = False
    th_time = 0

    start = time.time()
    while time.time() - start < dur:
        ret, frame = cap_down.read()
        if ret == True:
            cv2.imshow("show_r",frame)
            cv2.waitKey(1)
            print("FINDING")
            if pd_down(frame,crs):
                stop_it(stm32)
                th_time = time.time() - start
                fd = True
                break
        
        print(dur, ret)
        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=wh.hold)
        time.sleep(0.02)

    if fd:
        time.sleep(0.08)
        ret, frame = cap_down.read()
        if ret == True:
            if pd_down(frame,crs) == False:
                sscmd((0, -0.5,0,0,0,dt), stm32) # adjust
        sscmd((0, 0, 0, 0, 0, 0.1), stm32)   # Stop
        sscmd((0, 0, 0, -1, 0, dwns), stm32)  # Lower the actuator
        wh.hold = 1
        sscmd((0, 0, 0, 0, 1, 2), stm32)   # Activate suction
        sscmd((0, 0, 0, 1, 1, ups), stm32)   # Raise the actuator

        sscmd((-0.5,0,0,0,0,0.2),stm32) # little back

        # sscmd((0, -10), stm32)             # Move backward
        # sscmd((0, 0, 0, -1, 1, 2), stm32)   # Lower the actuator and keep suction on
        sscmd((0.05, -0.5, 0, 0, 1, th_time), stm32) # Return to the original position
        sscmd((0.8,0,0,0,0,0.5),stm32) # for more
        sscmd((-0.5,0,0,0,0,3),stm32) # go back
        sscmd((0,-20), stm32)
        return

    cmd = (0.02, -0.5, -0.0, 0, 0, ztim * 2)
    vx, vy, vrot, z_dir, fan, dur = cmd
    fd = False
    start = time.time()
    while time.time() - start < dur:
        ret, frame = cap_down.read()
        if ret == True:
            cv2.imshow("show_r",frame)
            cv2.waitKey(1)
            print(ret,"CATCHING")
            if pd_down(frame,crs):
                print("YES")
                stop_it(stm32)
                th_time = time.time() - start - ztim
                fd = True
                break
        
        print(dur,ret)
        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=wh.hold)
        time.sleep(0.02)

    if fd:
        time.sleep(0.08)
        ret, frame = cap_down.read()
        if ret == True:
            if pd_down(frame,crs) == False:
                sscmd((0,0.5,0,0,0,dt), stm32) # adjust
        sscmd((0, 0, 0, 0, 0, 0.1), stm32)   # Stop
        sscmd((0, 0, 0, -1, 0, dwns), stm32)  # Lower the actuator
        wh.hold = 1
        sscmd((0, 0, 0, 0, 1, 2), stm32)   # Activate suction
        sscmd((0, 0, 0, 1, 1, ups), stm32)   # Raise the actuator

        sscmd((-0.5,0,0,0,0,0.2),stm32) # little back
        # sscmd((0, -10), stm32)             # Move backward
        # sscmd((0, 0, 0, -1, 1, 2), stm32)   # Lower the actuator and keep suction on
        sscmd((0.02, 0.5, 0, 0, 1, th_time), stm32) # Return to the original position
        sscmd((0.02, -0.5, 0, 0, 1, 1), stm32) # Return to the original position
        sscmd((0.8,0,0,0,0,0.5),stm32) # for more
        sscmd((-0.5,0,0,0,0,3),stm32) # go back
        sscmd((0,-20), stm32)
        return

    stop_it(stm32)
    sscmd((-0.5,0,0,0,0,0.2),stm32) # little back
    sscmd((0.02, 0.5, 0, 0, 1, ztim), stm32)# Return to the original position
    sscmd((0.8,0,0,0,0,0.5),stm32) # for more
    sscmd((-0.5,0,0,0,0,3),stm32)
    sscmd((0,-20), stm32)


def catch_purple(stm32):
    g_left(gs_quarter, 3, stm32)
    adjust_face(2, 3, stm32)
    find_and_catch(1, stm32)


def put_down(stm32):
    sscmd((0, 20), stm32)
    sscmd((0, 20), stm32)
    sscmd((0, 0, 0, -1, 1, 2.5), stm32)   # Lower the actuator and keep suction on
    wh.hold = 0
    sscmd((0, 0, 0, 0, 0, 5), stm32)   # Wait for a moment
    sscmd((0, 0, 0, 1, 0, 3), stm32)   # Raise the actuator and turn off suction

def go_go_go(stm32):
    set_begin(stm32)
    sscmd((0,0,0,-1,0,2),stm32)
    sscmd((0, -20),stm32)
    go_from_begin(stm32)
    g_right(gs_quarter + 0.5 , 2 ,stm32, cp = 2) # watch ArUco2

    sscmd((-0.5,0,0,0,0,2),stm32)
    sscmd((-0.4,0,0,0,0,1),stm32)
    sscmd((-0.8,0,0,0,0,1),stm32)
    sscmd((0.5,0,0,0,0,3),stm32)

    # return
    ttt = 0
    while True:
        left_to_right(stm32, 6, 1000)
        g_left(gs_quarter , 4 ,stm32) # watch ArUco4

        sscmd((-0.5,0,0,0,0,2),stm32)
        sscmd((-0.4,0,0,0,0,1),stm32)
        sscmd((-0.8,0,0,0,0,1),stm32)
        sscmd((0.5,0,0,0,0,2),stm32)
        
        get_up_slope(stm32)

        # # adjust_clock(0.5, 4, stm32)
        # # adjust_face(1, 4, stm32)
        # # go_until(2, 4, 700, stm32)
        # # set_begin(stm32)
        # # sscmd((0,8),stm32) # for the last
        # # find_and_catch(0, stm32)
        # # sscmd((0, -20),stm32)

        turn_until(1, 4 ,stm32)
        while True:
            go_until(2, 4, 900, stm32)
            set_begin(stm32)
            sscmd((0,8),stm32) # for the last
            find_and_catch(0, stm32)
            sscmd((0, -20),stm32)
            # cap_down = wh.cap0
            # ret, frame = cap_down.read()
            # cv2.imshow("show_r",frame)
            break
            # if pd_down(frame, "orange", al = 1):
            #     print("GET THE BOX")
            #     break

        # adjust_face(1, 4, stm32)
        # adjust_clock(1, 4, stm32)
        # sscmd((0.5, 0,0,0,0,1), stm32)

        sscmd((-0.5,0,0,0,0,1.5), stm32)
        get_down_slope(stm32, td = -1)
        sscmd((-0.5,0,0,0,0,1),stm32)
        sscmd((-0.4,0,0,0,0,2),stm32)
        sscmd((0.5,0,0,0,0,1), stm32)
        g_left(gs_quarter * 2 + 0.1, 6, stm32) # turn back
        go_until(2, 6, 600, stm32)
        sscmd((0.5,0,0,0,1,1),stm32)
        sscmd((0.3, 0, 0, 0, 1, 2), stm32)
        sscmd((0.8,0,0,0,0,1),stm32) # for test and wait to del
        put_down(stm32)

        sscmd((-0.5,0,0,0,0,1.5), stm32)
        g_left(gs_quarter + 0.3, 5, stm32)

        ttt += 1


def main():

    # Mode selection

    stm32 = None

    lst_time = time.time()

    # Initialize communication

    stm32 = STM32Comm(port='/dev/ttyS0', baudrate=115200, enable=True)


    try:

        wh.cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
        wh.cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        wh.cap0.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        wh.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

        wh.cap4 = cv2.VideoCapture(4, cv2.CAP_V4L2)
        wh.cap4.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        wh.cap4.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        wh.cap4.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

        wh.cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
        wh.cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        wh.cap2.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        wh.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

        wh.K, wh.D = load_camera_params()

        print("All Right")
        
        while True:
            sss = input()
            cmd = tuple(map(float,sss.split()))
            sscmd(cmd,stm32)
            if(len(cmd) == 1):
                vl = cmd[0]
                if vl == -1:
                    set_begin(stm32)
                elif vl == 1:
                    turn_left(stm32)
                elif vl == 2:
                    turn_right(stm32)
                elif vl == 3:
                    go_and_get(stm32)
                elif vl == 4:
                    get_up_slope(stm32)
                elif vl == 5:
                    go_from_begin(stm32)
                elif vl == 6:
                    find_and_catch(0, stm32)
                elif vl == 7:
                    left_to_right(stm32, 8,650)
                elif vl == 8:
                    right_to_left(stm32, 3)
                elif vl == 9:
                    adjust_face(2, 4,stm32)
                elif vl == 10:
                    adjust_clock(2, 4,stm32)
                elif vl == 11:
                    w1 = float(input("time:"))
                    w2 = float(input("tb:"))
                    w3 = float(input("distance:"))
                    go_until(w1, w2, w3, stm32)
                elif vl == 12:
                    w1 = float(input("time:"))
                    w2 = float(input("tb:"))
                    g_left(w1,w2,stm32)
                elif vl == 13:
                    turn_until(2,4,stm32)

                elif vl == 21:
                    catch_purple(stm32)
                elif vl == 50 or vl == 51 or vl == 52:
                    gt_photo(int(vl - 50))

                elif vl == 61:
                    g_right(4*gs_quarter,5,stm32,cp = 1)
                elif vl == 62:
                    g_right(gs_quarter*2,2,stm32,cp = 2)
                elif vl == 63:
                    g_left(gs_quarter * 2 + 0.1, 6, stm32)

                elif vl == 100:
                    go_go_go(stm32)

        print("[DEBUG] End")
        return


    except KeyboardInterrupt:

        stm32.send_command(0, 0, 0, 0, 0, 0, 0)

        print("\n[SYSTEM] User interrupt")

        wh.cap0.release()
        wh.cap4.release()
        wh.cap2.release()

        if stm32 is not None:
            stm32.close()

        print("[DEBUG] End")

    finally:
        wh.cap0.release()
        wh.cap4.release()
        wh.cap2.release()

        if stm32 is not None:
            stm32.close()

        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

