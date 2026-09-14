"""

tot_detect.py - Master control program for ArUco navigation and block handling.

Uses two cameras (front and down) and communicates with STM32 via serial.

Step motors are now controlled via relative commands (send_move_relative).

Actuator is controlled in open-loop via direction in send_command (z_dir).

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

    """

    Return current control values.

    vy: forward speed, vx: lateral, w: angular, dn_v: actuator direction,

    nd_catch: suction on, nd_throw: suction off (release).

    """

    return wh.vy, wh.vx, wh.w, wh.dn_v, wh.nd_catch, wh.nd_throw


def _empty_result():
    return {'is_detected': False, 'position_mm': None, 'euler_deg': None,
            'distance_mm': None, 'reproj_error_px': None,
            'rvec': None, 'tvec': None, 'corners': None, 'detected_on': None}



def detect_all(frame=None, camera=None, marker_size_mm=None):

    """Detect all ArUco markers in the given frame."""

    if frame is None:

        temp_camera = None

        if camera is None:

            temp_camera = cv2.VideoCapture(CAM_FRONT)

        camera = temp_camera

        ret, frame = camera.read()

        if temp_camera is not None:

            temp_camera.release()

        if not ret:

            print("[WARNING] Failed to capture frame")

            return {mid: _empty_result() for mid in _POSE_FUNCS}

    K, D = load_camera_params()

    if K is None:

        return {mid: _empty_result() for mid in _POSE_FUNCS}

    results = {}

    for mid, func in _POSE_FUNCS.items():

        results[mid] = func(frame=frame, K=K, D=D, marker_size_mm=marker_size_mm)

    return results


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


def g_left(tim, tb, stm32, sl = 0):
    K, D = load_camera_params()
    start = time.time()
    while time.time() - start < tim:
        if sl == 1:
            stm32.send_command(0, 0, 0.5, x_mm=0, y_mm=0, z_dir=0, fan=0)
        else:
            stm32.send_command(0, 0, 0.8, x_mm=0, y_mm=0, z_dir=0, fan=0)
        cap = wh.cap4
        ret,frame = cap.read()
        cv2.imshow("show",frame)
        cv2.waitKey(1)
        result = _pose_id(tb, frame, K, D)
        if result['is_detected']:
            if result['euler_deg'][1] >= -1:
                break


        time.sleep(0.02)

    stm32.send_command(0, 0, 0, 0, 0, 0, 0)

def g_right(tim, tb, stm32, sl = 0):
    K, D = load_camera_params()
    start = time.time()
    while time.time() - start < tim:
        if sl == 1:
            stm32.send_command(0, 0, -0.5, x_mm=0, y_mm=0, z_dir=0, fan=0)
        else:
            stm32.send_command(0, 0, -0.8, x_mm=0, y_mm=0, z_dir=0, fan=0)
        cap = wh.cap4
        ret,frame = cap.read()
        cv2.imshow("show",frame)
        cv2.waitKey(1)
        result = _pose_id(tb, frame, K, D)
        if result['is_detected']:
            print(111)
            if result['euler_deg'][1] <= 1:
                break

        time.sleep(0.02)

    stm32.send_command(0, 0, 0, 0, 0, 0, 0)


def sscmd(cmd,stm32):

    if len(cmd) == 6:
        vx, vy, vrot, z_dir, fan, dur = cmd
        print(f"[DEBUG] Sending command: vx={vx}, vy={vy}, vrot={vrot}, z_dir={z_dir}, fan={fan}, dur={dur}")
        start = time.time()
        while time.time() - start < dur:

            stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=int(fan))

            time.sleep(0.02)


    elif len(cmd) == 2:
        vx,vy = cmd
        stm32.send_move_relative(vx, vy)

    elif len(cmd) == 3:
        typ,ta,tb = cmd
        if typ == 1:
            g_left(ta, tb, stm32)
        elif typ == 2:
            g_right(ta, tb, stm32)

    # Send relative move commands (separate)

    stm32.send_command(0, 0, 0, 0, 0, 0, 0)

def gt_photo(wh):
    print(wh)
    cap = wh.cap0
    if wh == 1:
        cap = wh.cap4
    elif wh == 2:
        cap = wh.cap2
    ret,frame = cap.read()
    cv2.imshow("show",frame)
    cv2.waitKey(1)

def adjust_clock(tim, tb, stm32):
    K, D = load_camera_params()
    cap = wh.cap4
    ret,frame = cap.read()
    cv2.imshow("show",frame)
    cv2.waitKey(1)
    result = _pose_id(tb, frame, K, D)
    if result['is_detected'] == False:
        print("not detect")
        return
    if result['euler_deg'][1] < 0:
        g_left(tim, tb, stm32,sl = 1)
    elif result['euler_deg'][1] > 0:
        g_right(tim, tb, stm32,sl = 1)

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
        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=int(fan))
        ret,frame = cap_down.read()
        if pd_down(frame,"orange"):
           fd = True
           break
        
        print(pd_down(frame,"orange"),dur)
        time.sleep(0.02)

    if fd == False:
        return
    
    sscmd((0,0,0,-1,0,3),stm32)
    sscmd((0,0,0,0,1,3),stm32)
    sscmd((0,0,0,1,1,3),stm32)
    sscmd((0,-30),stm32)
    sscmd((0,0,0,-1,1,2),stm32)

def get_up_slope(stm32):
    cmd = (0.8,0,0,0,0,2)
    sscmd(cmd,stm32)

quarter = 3.75

def turn_left(stm32):
    sscmd((0,0,1,0,0,quarter),stm32)

def turn_right(stm32): # for 1/4 round
    sscmd((0,0,-1,0,0,quarter),stm32)

def go_from_begin(stm32):
    K, D = load_camera_params()
    cmd = (0.5,0,0,0,0,2.5)
    vx, vy, vrot, z_dir, fan, dur = cmd
    fd = False
    start = time.time()
    cap = wh.cap4
    while time.time() - start < dur:
        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=int(fan))
        ret,frame = cap.read()
        cv2.imshow("show",frame)
        cv2.waitKey(1)
        result = _pose_id(1, frame, K, D)
        if result['is_detected']:
            print(111)
            if result['euler_deg'][1] <= 1:
                break
        time.sleep(0.02)

    g_right(quarter,5,stm32)

    stm32.send_command(0, 0, 0, 0, 0, 0, 0)


def left_to_right(stm32):
    cmd = (0.5,0,0,0,0,3)
    vx, vy, vrot, z_dir, fan, dur = cmd
    stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=int(fan))
    while True:
        ret,frame = wh.cap4.read()
        cv2.imshow("show",frame)
        cv2.waitKey(1)
        result = _pose_id(6, frame, wh.K, wh.D)
        if result['is_detected']:
            if result['position_mm'][1] * (-1) <= 1000:
                break
        time.sleep(0.02)

    stm32.send_command(0, 0, 0, 0, 0, 0, 0)

def find_and_catch(clr, stm32):
    # Move forward until the block is detected, then catch it
    ### have to guarantee that the catcher went ahead
    str = "orange"
    if clr == 1:
        str = "purple"
    ztim = 3
    ups = 3
    dwns = 2.7
    sscmd((0, 0, 0, 1, 0, ups), stm32)
    sscmd((0.5, 0, 0, 0, 0, 1), stm32)
    sscmd((0.35, 0, 0, 0, 0, 1), stm32)
    # sscmd((0, 30), stm32)             # Move ahead
    cmd = (0.05, 0.5, 0, 0, 0, ztim)
    vx, vy, vrot, z_dir, fan, dur = cmd
    fd = False
    th_time = 0
    cap_down = wh.cap0
    start = time.time()
    while time.time() - start < dur:
        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=int(fan))
        ret, frame = cap_down.read()
        cv2.imshow("show",frame)
        cv2.waitKey(1)
        if pd_down(frame,str):
            th_time = time.time() - start
            fd = True
            break
        
        print(pd_down(frame,str), dur)
        time.sleep(0.02)

    if fd:
        sscmd((0, 0, 0, 0, 0, 0.1), stm32)   # Stop
        sscmd((0, 0, 0, -1, 0, dwns), stm32)  # Lower the actuator
        sscmd((0, 0, 0, 0, 1, 2), stm32)   # Activate suction
        sscmd((0, 0, 0, 1, 1, ups), stm32)   # Raise the actuator
        sscmd((0, -10), stm32)             # Move backward
        sscmd((0, 0, 0, -1, 1, 2), stm32)   # Lower the actuator and keep suction on
        sscmd((0.05, -0.5, 0, 0, 1, th_time), stm32) # Return to the original position
        return

    cmd = (0.05, -0.5, 0, 0, 0, th_time + ztim)
    vx, vy, vrot, z_dir, fan, dur = cmd
    fd = False
    start = time.time()
    while time.time() - start < dur:
        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=int(fan))
        ret, frame = cap_down.read()
        if pd_down(frame,str):
            th_time = time.time() - start - th_time
            fd = True
            break
        
        print(pd_down(frame,str), dur)
        time.sleep(0.02)

    if fd:
        sscmd((0, 0, 0, 0, 0, 0.1), stm32)   # Stop
        sscmd((0, 0, 0, -1, 0, dwns), stm32)  # Lower the actuator
        sscmd((0, 0, 0, 0, 1, 2), stm32)   # Activate suction
        sscmd((0, 0, 0, 1, 1, ups), stm32)   # Raise the actuator
        sscmd((0, -10), stm32)             # Move backward
        sscmd((0, 0, 0, -1, 1, 2), stm32)   # Lower the actuator and keep suction on
        sscmd((0.05, 0.5, 0, 0, 1, th_time), stm32) # Return to the original position
        return


def go_go_go(stm32):
    go_from_begin(stm32)
    left_to_right(stm32)


def main():

    # Mode selection

    USE_VISUAL_CONTROL = False   # True: visual mode, False: manual debug

    stm32 = None

    lst_time = time.time()

    # Initialize communication

    stm32 = STM32Comm(port='/dev/ttyS0', baudrate=115200, enable=True)

    if USE_VISUAL_CONTROL:

        print("[SYSTEM] Starting mode: Visual automatic recognition")

        K, D = load_camera_params()

        if K is None:

            print("[ERROR] Failed to load camera parameters")

            return

    else:

        print("[SYSTEM] Starting mode: Manual debug")



    try:

        if USE_VISUAL_CONTROL:
            old_dn = True

        else:

            wh.cap0 = cv2.VideoCapture(0)
            wh.cap4 = cv2.VideoCapture(4)
            wh.cap2 = cv2.VideoCapture(2)
            wh.K, wh.D = load_camera_params()
            
            while True:
                sss = input()
                cmd = tuple(map(float,sss.split()))
                sscmd(cmd,stm32)
                if(len(cmd) == 1):
                    vl = cmd[0]
                    if vl == -1:
                        break
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
                        left_to_right(stm32)
                    elif vl == 10 or vl == 11 or vl == 12:
                        gt_photo(int(vl - 10))

            sscmd((0,0,0,0,0,10)) # stop

            wh.cap0.release()
            wh.cap4.release()
            wh.cap2.release()

            print("[DEBUG] Starting manual command sequence...")
            return

        
            commands = [ # front,left,un-clock,up_only_f1_0_1,air_open_01,time
                (0 , 0 , 1 , 0 , 0 , 4.6),
                (-1,-1,-1),
            ]
            for cmd in commands:
                if len(cmd) == 6:
                    vx, vy, vrot, z_dir, fan, dur = cmd

                    print(f"[DEBUG] Sending command: vx={vx}, vy={vy}, vrot={vrot}, z_dir={z_dir}, fan={fan}, dur={dur}")
                    start = time.time()
                    while time.time() - start < dur:
                        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=z_dir, fan=fan)
                        time.sleep(0.02)

                elif len(cmd) == 2:
                    vx,vy = cmd
                    stm32.send_move_relative(vx, vy)
                
                elif len(cmd) == 3:
                    pdt,_,_ = cmd
                    if pdt == -1:
                        break

            # Send relative move commands (separate)

            print("[DEBUG] Sending relative move: X+10mm, Y-5mm")

            #stm32.send_move_relative(0.0, 10.0)#platform: left_and_right,front

            time.sleep(100.0)   # wait for movement (depends on speed)

            print("[DEBUG] Finished")


    except KeyboardInterrupt:

        print("\n[SYSTEM] User interrupt")

    finally:

        # if cap_front is not None and cap_front.isOpened():

        #     cap_front.release()

        # if cap_down is not None and cap_down.isOpened():

        #     cap_down.release()

        if stm32 is not None:

            stm32.close()

        cv2.destroyAllWindows()



if __name__ == "__main__":

    main()

