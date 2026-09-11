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

    print("[ERROR] stm32_comm.py not found. Please ensure the communication module is in the same directory.")

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

CAM_FRONT = 2   # Forward camera



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

mx_high = 100  # Max arm height (not used in open-loop)



def getspeed():

    """

    Return current control values.

    vy: forward speed, vx: lateral, w: angular, dn_v: actuator direction,

    nd_catch: suction on, nd_throw: suction off (release).

    """

    return wh.vy, wh.vx, wh.w, wh.dn_v, wh.nd_catch, wh.nd_throw



def giv_high(high1):

    # Kept for compatibility, but actuator is open-loop, so this is unused.

    wh.dn_high = high1



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



def to_area(tmp):

    while tmp < 0:

        tmp += 2 * math.pi

    while tmp >= 2 * math.pi:

        tmp -= 2 * math.pi

    return tmp



def gt_adjust_w(tw, tdir, rs):

    """Fine-tune angular velocity based on detected markers."""

    cnt = 0

    tmp = 0.0

    pi = math.pi

    if rs[1] is not None and rs[1]['is_detected']:

        cnt += 1; tmp += to_area(rs[1]['euler_deg'][1] - pi / 2)

    if rs[2] is not None and rs[2]['is_detected']:

        cnt += 1; tmp += to_area(rs[2]['euler_deg'][1])

    if rs[3] is not None and rs[3]['is_detected']:

        cnt += 1; tmp += to_area(rs[3]['euler_deg'][1] - pi / 2)

    if rs[4] is not None and rs[4]['is_detected']:

        cnt += 1; tmp += to_area(rs[4]['euler_deg'][1])

    if rs[5] is not None and rs[5]['is_detected']:

        cnt += 1; tmp += to_area(rs[5]['euler_deg'][1] + pi / 2)

    if rs[6] is not None and rs[6]['is_detected']:

        cnt += 1; tmp += to_area(rs[6]['euler_deg'][1] + pi)



    if cnt > 0:

        tw = tmp / cnt

        wh.w = tw

        if tw - 0.01 <= tdir <= tw + 0.01:

            return 0

        elif tw < tdir:

            return 1

        else:

            return -1

    return 0


def sscmd(cmd,stm32):

    if len(cmd) == 6:

        vx, vy, vrot, z_dir, fan, dur = cmd
        print(f"[DEBUG] Sending command: vx={vx}, vy={vy}, vrot={vrot}, z_dir={z_dir}, fan={fan}, dur={dur}")
        start = time.time()
        while time.time() - start < dur:

            stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=int(fan))

            #cap_down.release()
            time.sleep(0.05)


    elif len(cmd) == 2:

        vx,vy = cmd

        stm32.send_move_relative(vx, vy)

    

    elif len(cmd) == 3:

        pdt,_,_ = cmd

        if pdt == -1:

            return

            # wh.stat = False # need to stop the robot
    # Send relative move commands (separate)

    stm32.send_command(0, 0, 0, 0, 0, 0, 0)

def gt_photo(wh):
    cap = cv2.VideoCapture(wh)
    ret,frame = cap.read()
    cv2.imshow("show",frame)
    cap.release()

def go_and_get(stm32):# keep the stuff on the top
    # front,left,un-clock,up_only_f1_0_1,air_open_01,time
    cmd = (0.5,0,0,0,0,3.5)
    vx, vy, vrot, z_dir, fan, dur = cmd
    fd = False
    sscmd((0,0,0,1,0,3),stm32)
    start = time.time()
    while time.time() - start < dur:
        stm32.send_command(vx, vy, vrot, x_mm=0, y_mm=0, z_dir=int(z_dir), fan=int(fan))
        cap_down = cv2.VideoCapture(0)
        ret,frame = cap_down.read()
        if pd_down(frame,"orange"):
           cap_down.release()
           fd = True
           break
        
        print(pd_down(frame,"orange"),dur)
        cap_down.release()
        time.sleep(0.05)

    if fd == False:
        return
    
    sscmd((0,0,0,-1,0,3),stm32)
    sscmd((0,0,0,0,1,3),stm32)
    sscmd((0,0,0,1,1,3),stm32)
    sscmd((0,-20),stm32)
    sscmd((0,0,0,-1,1,2),stm32)

def get_on_top(stm32):
    cmd = (0.8,0,0,0,0,2)
    vx, vy, vrot, z_dir, fan, dur = cmd

def turn_left(stm32):
    sscmd((0,0,1,0,0,3.65),stm32)

def turn_right(stm32): # for 1/4 round
    sscmd((0,0,-1,0,0,3.65),stm32)

def main():

    # ============================================================

    # Mode selection

    # ============================================================

    USE_VISUAL_CONTROL = False   # True: visual mode, False: manual debug



    cap_front = None

    cap_down = None

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



        cap_front = cv2.VideoCapture(CAM_FRONT)

        cap_down = cv2.VideoCapture(CAM_DOWN)

        if not cap_front.isOpened():

            print(f"[ERROR] Cannot open front camera (index {CAM_FRONT})")

            return

        if not cap_down.isOpened():

            print(f"[ERROR] Cannot open down camera (index {CAM_DOWN})")

            return



        cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cap_down.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        cap_down.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    else:

        print("[SYSTEM] Starting mode: Manual debug")



    try:

        if USE_VISUAL_CONTROL:
            old_dn = True

        else:

            while True:

                sss = input()

                cmd = tuple(map(float,sss.split()))

                sscmd(cmd,stm32)

                # cap_down = cv2.VideoCapture(0)

                # ret,frame = cap_down.read()

                # print(pd_down(frame,"orange"))

                # cap_down.release()

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

            # === Manual debug mode ===

            sscmd((0,0,0,0,0,10)) # stop

            print("[DEBUG] Starting manual command sequence...")

            return

            # Example sequence: move chassis, relative step, actuator

            # move,xipan_move and stop in commands

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

                        time.sleep(0.05)

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

        if cap_front is not None and cap_front.isOpened():

            cap_front.release()

        if cap_down is not None and cap_down.isOpened():

            cap_down.release()

        if stm32 is not None:

            stm32.close()

        cv2.destroyAllWindows()



if __name__ == "__main__":

    main()

