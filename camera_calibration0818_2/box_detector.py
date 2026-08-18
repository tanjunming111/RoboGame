import cv2
import numpy as np
import math

class is_down:
    def __init__(self):
        self.ys = False

dn = is_down()

# 填入第二步调好的阈值
LOWER_ORANGE = np.array([9, 30, 115])
UPPER_ORANGE = np.array([35, 170, 255])
LOWER_PURPLE = np.array([9, 30, 115])
UPPER_PURPLE = np.array([35, 170, 255])

# 形态学操作的核大小，用于去噪和填补空洞
MORPH_KERNEL_SIZE = 5

# 高斯模糊的参数
blur_size = 5

# Canny
low_t=40
high_t=80

# 霍夫变换
hough_thresh = 30
min_len = 30
max_gap = 20

def solve_pd(mask):
    mask_in = mask.copy()
    h, w = mask_in.shape
    cx, cy = int(w * 0.5), int(h * 0.4)  # 圆心位置（中间偏上）
    RR = 80 # 半径

    # 1.生成圆形掩码
    circle_mask = np.zeros((h, w),dtype = np.uint8)
    cv2.circle(circle_mask, (cx,cy), RR, 255, -1)

    # 2.只保留圆形区域
    roi = cv2.bitwise_and(mask_in, circle_mask)

    # 3.统计占比
    white_in = cv2.countNonZero(roi)
    total_in = cv2.countNonZero(circle_mask)

    ratio = white_in / total_in
    print(ratio)
    ret = False
    if ratio >= 0.95:
        # dn.ys = True
        ret = True
    else:
        # dn.ys = False
        ret = False

    cv2.circle(mask_in, (cx, cy), RR, (255, 255, 255), 2)
    cv2.imshow("circle", mask_in)
    return ret

def pd_down(frame, str):
    # ===== 第1步：高斯模糊（降噪）=====
    blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 1)

    # ===== 第2步：Canny 检测
    canny_direct = cv2.Canny(blurred, low_t, high_t)
    
    # BGR -> HSV
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

    # 生成掩码，白色为观察范围
    if str == "orange":
        mask=cv2.inRange(hsv,LOWER_ORANGE,UPPER_ORANGE)
    elif str == "purple":
        mask=cv2.inRange(hsv,LOWER_PURPLE,UPPER_PURPLE)

    # 创建形态学核（圆形核）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 开运算：去除噪声
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 闭运算：填补空洞
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=2)

    ker = np.ones((3,3),np.uint8) #生成腐蚀/膨胀核 
    erosion = cv2.erode(mask_close,ker,iterations = 1) #腐蚀
    ker = np.ones((5,5),np.uint8) #生成腐蚀/膨胀核 
    mask = cv2.dilate(erosion,ker,iterations = 1) #膨胀

    # Canny 检测
    canny=cv2.Canny(blurred,low_t,high_t)
    canny=cv2.bitwise_and(canny,mask)

    ker = np.ones((2,2),np.uint8) #生成腐蚀/膨胀核 
    canny = cv2.dilate(canny,ker,iterations = 1) #膨胀

    # 霍夫变换
    lines_raw = cv2.HoughLinesP(canny,
                            rho=1,              # 距离分辨率（像素）
                            theta=np.pi / 180,  # 角度分辨率（弧度）
                            threshold=hough_thresh,  # 投票阈值，越大越严格
                            minLineLength=min_len,    # 最短线段长度
                            maxLineGap=max_gap)       # 允许的最大间隙
    line_img = frame.copy()
    if lines_raw is not None:
        for line in lines_raw:
            x1,y1,x2,y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return solve_pd(mask)

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== 第1步：高斯模糊（降噪）=====
        blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 1)

        # ===== 第2步：Canny 检测
        canny_direct = cv2.Canny(blurred, low_t, high_t)
        
        # BGR -> HSV
        hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

        # 生成掩码，白色为观察范围
        mask=cv2.inRange(hsv,LOWER_ORANGE,UPPER_ORANGE)

        # 创建形态学核（圆形核）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 开运算：去除噪声
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 闭运算：填补空洞
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=2)

        ker = np.ones((3,3),np.uint8) #生成腐蚀/膨胀核 
        erosion = cv2.erode(mask_close,ker,iterations = 1) #腐蚀
        ker = np.ones((5,5),np.uint8) #生成腐蚀/膨胀核 
        mask = cv2.dilate(erosion,ker,iterations = 1) #膨胀

        # Canny 检测
        canny=cv2.Canny(blurred,low_t,high_t)
        canny=cv2.bitwise_and(canny,mask)

        ker = np.ones((2,2),np.uint8) #生成腐蚀/膨胀核 
        canny = cv2.dilate(canny,ker,iterations = 1) #膨胀

        # 霍夫变换
        lines_raw = cv2.HoughLinesP(canny,
                                rho=1,              # 距离分辨率（像素）
                                theta=np.pi / 180,  # 角度分辨率（弧度）
                                threshold=hough_thresh,  # 投票阈值，越大越严格
                                minLineLength=min_len,    # 最短线段长度
                                maxLineGap=max_gap)       # 允许的最大间隙
        line_img = frame.copy()
        if lines_raw is not None:
            for line in lines_raw:
                x1,y1,x2,y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        pd_down(frame, "orange")

        # cv2.imshow("frame",frame)
        # cv2.imshow("show",mask)
        # cv2.imshow("canny",canny)
        # cv2.imshow("hough",line_img)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows

if __name__ == "__main__":
    main()
