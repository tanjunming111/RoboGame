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


def sscmd(cmd):
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
            return
            # wh.stat = False # need to stop the robot
        
    # Send relative move commands (separate)


def go_and_get():
    sscmd((0.5,0,0,0,0,2))

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

            while True:

                now = time.time()

                ret_f, frame_front = cap_front.read()

                ret_d, frame_down = cap_down.read()

                if not ret_f or not ret_d:

                    print("[WARNING] Failed to read one of the cameras")

                    break



                # 1. ArUco detection

                results = detect_all(frame=frame_front)

                rs = [None] * 7

                for id, r in sorted(results.items()):

                    rs[id] = r



                # 2. Display (undistort and draw)

                frame_undist = cv2.undistort(frame_front, K, D)

                display = frame_undist.copy()

                for mid, r in sorted(results.items()):

                    if r['is_detected']:

                        pts = r['corners'].astype(int)

                        for k in range(4):

                            cv2.line(display, tuple(pts[k]), tuple(pts[(k + 1) % 4]), (0, 255, 0), 2)

                        cv2.putText(display, f"ID={mid}", (int(pts[0][0]), int(pts[0][1]) - 10),

                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



                # 3. State machine logic (unchanged from original)

                dt = now - lst_time

                wh.x += (wh.vx * 1000 * math.cos(wh.tn_ag) + wh.vy * 1000 * math.sin(wh.tn_ag)) * dt

                wh.y += (wh.vx * 1000 * math.sin(wh.tn_ag) + wh.vy * 1000 * math.cos(wh.tn_ag)) * dt

                wh.tn_ag += wh.w * dt



                if not wh.rot:

                    wh.w = gt_adjust_w(wh.w, wh.nd_dir * math.pi / 2, rs) * 0.1



                # Step 0: Initial localization

                if wh.step == 0:

                    if not wh.bg and rs[1] is not None and rs[1]['is_detected']:

                        wh.y = -rs[1]['position_mm'][0]

                        wh.x = -rs[1]['position_mm'][2]

                        wh.bg = True

                        wh.vy = BASE_SPEED

                    if rs[1] is not None and rs[1]['is_detected']:

                        wh.y = -rs[1]['position_mm'][0]

                        wh.x = -rs[1]['position_mm'][2]

                    if wh.bg and wh.y >= 0:

                        wh.vy = 0

                        wh.step = 1

                        wh.rot = True

                        wh.rot_bg = now

                        wh.w = math.pi / 2 / 2

                        wh.nd_dir = 1



                # Step 1: Turn right

                elif wh.step == 1:

                    if now - wh.rot_bg >= 1 or (rs[5] is not None and rs[5]['is_detected'] and rs[5]['euler_deg'][1] >= -1):

                        wh.rot = False

                        wh.w = 0

                        wh.step = 2

                        wh.vy = BASE_SPEED

                        wh.ndy = 3100



                # Step 2: Move forward to target

                elif wh.step == 2:

                    if rs[6] is not None and rs[6]['is_detected']:

                        wh.ndy = wh.y - rs[6]['position_mm'][0]

                    if wh.y >= wh.ndy or (rs[5] is not None and rs[5]['is_detected'] and rs[5]['position_mm'][2] <= 900):

                        wh.vy = 0

                        wh.step = 3

                        wh.rot = True

                        wh.rot_bg = now

                        wh.w = -math.pi / 2 / 2

                        wh.nd_dir = 0



                # Step 3: Turn forward

                elif wh.step == 3:

                    if now - wh.rot_bg >= 1 or (rs[4] is not None and rs[4]['is_detected'] and rs[4]['euler_deg'][1] <= 1):

                        wh.rot = False

                        wh.w = 0

                        wh.step = 4

                        wh.vy = BASE_SPEED



                # Step 4: Move forward to marker 4

                elif wh.step == 4:

                    if rs[4] is not None and rs[4]['is_detected'] and -rs[4]['position_mm'][2] <= 150:

                        wh.vy = 0

                        wh.step = 5

                        wh.vx = -BASE_SPEED



                # Step 5: Move left to find block

                elif wh.step == 5:

                    if pd_down(frame_down, "orange"):

                        wh.vx = 0

                        wh.step = -1

                        wh.nxt = 7

                        wh.catch_ps = -1

                        wh.dn_v = 1   # Extend actuator (open-loop direction)

                    elif wh.x <= 2350:

                        wh.vx = BASE_SPEED

                        wh.step = 6



                # Step 6: Move right to find block

                elif wh.step == 6:

                    if pd_down(frame_down, "orange"):

                        wh.vx = 0

                        wh.step = -1

                        wh.nxt = 7

                        wh.catch_ps = 1

                        wh.dn_v = 1   # Extend

                    elif wh.x >= 3850:

                        wh.vx = -BASE_SPEED

                        wh.step = 7



                # Step -1: Lower and grab (open-loop, so just extend until stopped)

                elif wh.step == -1:

                    # Assume lowering completes when commanded; we just set dn_v=1 already

                    # Wait a fixed time? For simplicity, we transition based on time or other condition.

                    # In open-loop we don't know position, so use time delay.

                    if wh.dn_high == 0:  # This variable not used now, but keep transition

                        wh.dn_v = 0      # Stop actuator

                        wh.nd_catch = True

                        wh.step = -2

                        wh.slp_time = now + 1

                    # For open-loop, we might need to add a timeout instead.

                    # Here we assume a simple switch after some time has passed.

                    # We'll leave as is but note it may need adjustment.



                # Step -2: Wait for grab

                elif wh.step == -2:

                    if now >= wh.slp_time:

                        wh.nd_catch = False

                        wh.step = -3

                        wh.dn_v = -1   # Retract actuator



                # Step -3: Raise (open-loop: retract until stopped)

                elif wh.step == -3:

                    # We cannot know when it reaches top without feedback,

                    # so we wait a fixed time (e.g., 2 seconds) then stop.

                    if now - wh.slp_time > 2:   # crude timeout

                        wh.dn_v = 0

                        wh.step = wh.nxt



                # Step 7: Return to marker 4

                elif wh.step == 7:

                    wh.vx = -wh.catch_ps * BASE_SPEED

                    if rs[4] is not None and rs[4]['is_detected'] and rs[4]['position_mm'][0] >= 0:

                        wh.vx = 0

                        wh.step = 10

                        wh.rot = True

                        wh.rot_bg = now

                        wh.w = -math.pi / 2 / 2

                        wh.nd_dir = 2



                # Step 10: Turn backward

                elif wh.step == 10:

                    if now - wh.rot_bg >= 2 or (rs[6] is not None and rs[6]['is_detected'] and rs[6]['euler_deg'][1] <= 1):

                        wh.rot = False

                        wh.w = 0

                        wh.step = 11

                        wh.vy = BASE_SPEED



                # Step 11: Move backward to drop point

                elif wh.step == 11:

                    if rs[6] is not None and rs[6]['is_detected'] and -rs[6]['position_mm'][2] <= 150:

                        wh.vy = 0

                        wh.step = -11

                        wh.dn_v = 1   # Extend to lower

                        wh.nxt = 20



                # Step -11: Lower to release

                elif wh.step == -11:

                    # Wait fixed time then stop and release

                    if now - wh.slp_time > 2:   # crude timeout

                        wh.dn_v = 0

                        wh.nd_throw = True

                        wh.step = -12

                        wh.slp_time = now + 7



                # Step -12: Wait for release

                elif wh.step == -12:

                    if now >= wh.slp_time:

                        wh.nd_throw = False

                        wh.step = -13

                        wh.dn_v = -1   # Retract



                # Step -13: Raise

                elif wh.step == -13:

                    if now - wh.slp_time > 2:   # crude timeout

                        wh.dn_v = 0

                        wh.step = wh.nxt



                # Step 20: Final reset

                elif wh.step == 20:

                    wh.vx = BASE_SPEED



                # 4. Prepare data to send

                c_vy, c_vx, c_w, c_dn_v, c_nd_catch, c_nd_throw = getspeed()



                # Map actuator direction from float to int (-1,0,1)

                actuator_dir = 0

                if c_dn_v > 0.1:

                    actuator_dir = 1

                elif c_dn_v < -0.1:

                    actuator_dir = -1



                fan_state = 1 if c_nd_catch else (0 if c_nd_throw else 0)



                # 5. Send command (no absolute step movement)

                stm32.send_command(

                    vx=c_vy,

                    vy=c_vx,

                    vrot=c_w,

                    x_mm=0.0,   # ignored

                    y_mm=0.0,   # ignored

                    z_dir=actuator_dir,

                    fan=fan_state

                )



                # Note: Step motors are not moved in this visual logic (assuming fixed positions).

                # If relative movement is needed, call stm32.send_move_relative(dx, dy) at appropriate steps.



                # 6. Display

                cv2.putText(display, f"Step:{wh.step} X:{wh.x:.0f} Y:{wh.y:.0f}", (10, 30),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.putText(display, f"Vx:{c_vx:.2f} Vy:{c_vy:.2f} ActDir:{actuator_dir}", (10, 60),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.imshow("Front Camera - ArUco Navigation", display)

                cv2.imshow("Down Camera - Block Detection", frame_down)



                if cv2.waitKey(30) & 0xFF == ord('q'):

                    break

                lst_time = now



        else:

            # === Manual debug mode ===

            print("[DEBUG] Starting manual command sequence...")

            # Example sequence: move chassis, relative step, actuator

            # move,xipan_move and stop in commands

            commands = [ # front,left,un-clock,up_only_f1_0_1,air_open_01,time

                (0 , 0 , 1 , 0 , 0 , 4.6),

                (-1,-1,-1),

                (0,0,0,1,0,2.5),

                (0.5,0,0,0,0,3.0),

                (0.3,0,0,0,0,3),

                (0,30),

                (0,0,0,-1,0,3.0),

                (0,0,0,0.0,1,3.0),

                (0,0,0,1,1,2.5),

                (0,-30)

                (-0.5,0,0,0.0,1,1.5),

                (-0.3,0,0,0.0,1,1.5),

                (0,0,1,0,1,4.6),

                (0.5,0,0,0,1,5.0),

                (0.3,0,0,0,1,5),

                (0,0,0,-1,1,2.5),

                (0,0,0,0,0,3),

                (0,0,0,1,0,2.5),

                (0,0,0,0,0,100),

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

