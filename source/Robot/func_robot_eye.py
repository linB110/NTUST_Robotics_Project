import cv2 
from cv2 import SOLVEPNP_ITERATIVE
import numpy as np
import re
import socket
import serial 
import time
import math
# COM = 'COM8'

TCP_IP = '192.168.0.1' #  Robot IP address. Start the TCP server from the robot before starting this code
TCP_PORT = 3000  #  Robot Port
BUFFER_SIZE = 1024  #  Buffer size of the channel, probably 1024 or 4096
global c
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #  Initialize the communication the robot through TCP as a client, the robot is the server. 
                                                        #  Connect the ethernet cable to the robot electric box first
c.connect((TCP_IP, TCP_PORT)) 

color_ranges1 = {'red' : ([0, 234, 71], [179, 255, 129]), 'yellow' : ([6, 125, 85], [142, 255, 255]), 
               'green' : ([47, 177, 21], [78, 255, 67]), 'purple' : ([73, 72, 62], [131, 160, 255]),
               'black' : ([50, 152, 0], [123, 255, 7])}

color_ranges2 = {'red' : ([0, 201, 65], [179, 255, 167]), 'yellow' : ([10, 138, 0], [44, 255, 255]), 
               'green' : ([36, 51, 0], [76, 255, 68]), 'purple' : ([104, 42, 79], [132, 69, 119]),
               'black' : ([83, 0, 0], [139, 255, 64])}

# camera parameters
sur_cam_mat = np.array([[677.88763981, 0, 320.19544367],
                        [0, 677.12555096, 233.44767356],
                        [0, 0, 1]])

sur_cam_dist = np.array([[0.14938788 ,-0.42509082 ,-0.00487487 , 0.0031457 , -1.06429]])

robot_cam_mat = np.array([[666.48396616, 0, 303.46613738],
                          [0, 666.04095729 ,239.10542119],
                          [0, 0, 1]])

robot_cam_dist = np.array([[5.63312812e-02, -3.48297009e-01, -1.26193791e-03, 2.56904212e-04, 6.09962652e-01]])

object_points = np.array([[0, 0, 0], [50, 0, 0], [50, 50, 0], [0, 50, 0]], dtype=np.float32) # 左上 右上 右下 左下

cam2base_mat = np.array([[0, 1, 0, 150],
                         [1, 0, 0, 580],
                         [0, 0, -1, 860],
                         [0, 0, 0, 1]])

cube = np.array([[25], [25], [0], [1]])

hand_eye_mat = np.array([[0.041316045701030646, -0.9979482595977749, -0.04891068935726167, -72.92001196596115],
                         [0.9988713396092066, 0.04010715660694353, 0.025445292221058645, -1.1063827089040004],
                         [-0.02343141640915701, -0.04990678465577907, 0.9984789840404167, 80.03635362161896],
                         [0, 0, 0, 1]])

tx = np.array([[1, 0, 0, -100],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

def cam_pict (index): # 相機拍照
    cap = cv2.VideoCapture(index) # number of surrounding camera
    if not cap.isOpened():
        print("無法打開 Webcam !")
        exit()

    else:
        ret, frame = cap.read()

        if ret:
            # 保存照片到指定位置
            save_path = F"C://Users//asus//OneDrive//desktop//python training//{index}.jpg"  # 指定保存位置和文件名
            cv2.imwrite(save_path, frame)
            print("照片已保存到:", save_path)
        else:
            print("無法捕捉照片！")

        # 释放 Webcam
        cap.release()
        cv2.destroyAllWindows()

def contour (): #從相機所拍攝照片進行角點偵測

    img = cv2.imread('0.jpg')
    copy = img.copy()
    hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    color_loc1 = {}

    for color_name, (lower, upper) in color_ranges1.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        blur = cv2.GaussianBlur(mask,(9, 9), 0)
        (cnts, _) = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"]) #m10:一階水平矩、m00:物體的面積
                cy = int(M["m01"]/M["m00"]) #m01:一階垂直矩
                cv2.circle(copy, (cx, cy), 3, (0, 0, 255), -1)
            height, width = copy.shape[:2]
            back = np.zeros((height, width), dtype = np.uint8)
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx_C = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(back, [approx_C], -1, (255, 255, 255), 3)
            corners = cv2.goodFeaturesToTrack(back, maxCorners = 4, qualityLevel = 0.03, minDistance = 10)
            if corners is not None:
                corners = np.int0(corners)
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(copy, (x, y), 3, (0, 255, 0), -1)
                    point_coor = '(' + str(x) + ',' + str(y) +')'
                    cv2.putText(copy, point_coor, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    # 判斷角點位置
                    if x < cx and y < cy: # 左上
                        left_top = (x, y)
                    elif x < cx and y > cy: # 左下
                        left_down = (x, y)
                    elif x > cx and y < cy: # 右上
                        right_top = (x, y)
                    elif x > cx and y > cy: # 右下
                        right_down = (x, y)

            color_loc1[color_name] = (left_top, right_top, right_down, left_down) #將找到的4個角點新增入字典
            text_coor = '(' + str(cx) + ',' + str(cy) +')'
            text_col = color_name
            cv2.putText(copy, text_coor, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(copy, text_col, (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    print(color_loc1)
    cv2.imshow('contour', copy)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    return color_loc1
    
def bot_contour (color): #從手臂相機所拍攝照片進行角點偵測
    img = cv2.imread('1.jpg')
    copy = img.copy()
    hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    color_loc2 = []

    lower, upper = color_ranges2[color]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    blur = cv2.GaussianBlur(mask,(9, 9), 0)
    (cnts, _) = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"]) #m10:一階水平矩、m00:物體的面積
            cy = int(M["m01"]/M["m00"]) #m01:一階垂直矩
            cv2.circle(copy, (cx, cy), 3, (0, 0, 255), -1)
        height, width = copy.shape[:2]
        back = np.zeros((height, width), dtype = np.uint8)
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx_C = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(back, [approx_C], -1, (255, 255, 255), 3)
        corners = cv2.goodFeaturesToTrack(back, maxCorners = 4, qualityLevel = 0.03, minDistance = 10)
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(copy, (x, y), 3, (0, 255, 0), -1)
                point_coor = '(' + str(x) + ',' + str(y) +')'
                cv2.putText(copy, point_coor, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    
                # 判斷角點位置
                if x < cx and y < cy: # 左上
                    left_top = (x, y)
                elif x < cx and y > cy: # 左下
                    left_down = (x, y)
                elif x > cx and y < cy: # 右上
                    right_top = (x, y)
                elif x > cx and y > cy: # 右下
                    right_down = (x, y)

        color_loc2 = [left_top, right_top, right_down, left_down] #將找到的角點新增入list
        text_coor = '(' + str(cx) + ',' + str(cy) +')'
        text_col = color
        cv2.putText(copy, text_coor, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(copy, text_col, (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        #print(color_name + " ： " + '(' + str(cx) + ',' + str(cy) + ')')    

    print(color_loc2)
    cv2.imshow('contour', copy)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    return color_loc2

def sur_cam_pnp (color_loc1, color): # 進行第一次pnp

    sur_image_points = np.array(color_loc1[color], dtype=np.float32)
    print(sur_image_points)
    flags = cv2.SOLVEPNP_SQPNP
    success, rotation_vector, translation_vector = cv2.solvePnP(object_points, sur_image_points, sur_cam_mat, sur_cam_dist, flags)

    if success:
        # 計算所得旋轉向量、平移向量
        print("計算所得旋轉向量：")
        print(rotation_vector*180/np.pi)
        print("計算所得平移向量：")
        print(translation_vector)
        print('------------------')
    
    else:
        print("PnP求解失败。")

    R, _ = cv2.Rodrigues(rotation_vector)
    T = np.vstack([np.hstack([R, translation_vector]), [0, 0, 0, 1]])

    print("方塊至相機轉移矩陣 T：\n", T)
    return T

def bot_pnp(color_loc2): # 進行第二次pnp

    bot_image_points = np.array(color_loc2, dtype=np.float32)
    flags = cv2.SOLVEPNP_SQPNP
    success, rotation_vector, translation_vector = cv2.solvePnP(object_points, bot_image_points, robot_cam_mat, robot_cam_dist, flags)

    if success:
        # 計算所得旋轉向量、平移向量
        print("計算所得旋轉向量：")
        print(rotation_vector*180/np.pi)
        print("計算所得平移向量：")
        print(translation_vector)
        print('------------------')
    
    else:
        print("PnP求解失败。")

    R2, _ = cv2.Rodrigues(rotation_vector)
    T2 = np.vstack([np.hstack([R2, translation_vector]), [0, 0, 0, 1]])

    print("方塊至手臂相機轉移矩陣 T：\n", T2)
    return T2

def robot_move(x, y, z, rx, ry, rz, c):
    mesg = f'{x},{y},{z},{rx},{ry},{rz}'
    c.send(bytes(mesg, "utf-8"))

def receive(c):
    data = c.recv(1024).decode()
    data = re.sub(',', ' ', data)
    pos = data.split()
    pos = list(map(float, pos))  # [x, y, z, rx - psi, ry - theta, rz - phi]
    return pos

def move_1(T): # 進行第一次pnp結果移動--移動至方塊附近

    tran = np.dot(cam2base_mat,T)
    cord1 = np.dot(tran, cube)
    cord_x = cord1[0][0]
    cord_y = cord1[1][0]
    print('TCP結果為')
    print(cord_x)
    print(cord_y)

    robot_move(cord_x, cord_y, 500, -180, 0, -180, c)  #  傳送座標
    # rx = -180  z_min = 400
    rPos = receive(c) #接收座標
    print(rPos)
    #c.close()  # 關閉連接(不確並需不需要)
    return rPos

def move_2(T2, rPos, color_loc2): # 進行第二次pnp結果移動--移動至方塊上方

    x, y, z, rx, ry, rz = rPos[:6]
    sa = np.sin(rx*np.pi/180)
    ca = np.cos(rx*np.pi/180)
    sb = np.sin(ry*np.pi/180)
    cb = np.cos(ry*np.pi/180)
    sg = np.sin(rz*np.pi/180)
    cg = np.cos(rz*np.pi/180)
    
    # 手臂姿態
    tran_mat = np.array([[cg*cb, -sg*ca+cg*sb*sa, sg*sa+cg*sb*ca, x],
                         [sg*cb, cg*ca+sg*sb*sa, -cg*sa+sg*sb*ca, y],
                         [-sb, cb*sa, cb*ca, z],
                         [0, 0, 0, 1]])
    
    cal2 = np.dot(tran_mat, hand_eye_mat)
    cal3 = np.dot(cal2, T2)
    cord2 = np.dot(cal3, cube)
    cord_x2 = cord2[0][0]
    cord_y2 = cord2[1][0]
    cord_z2 = 350
    left_top, right_top = color_loc2[:2]
    dx = right_top[0] - left_top[0]
    dy = right_top[1] - left_top[1]
    rz2 = math.degrees(math.atan2(dy, dx))
    print(rz2)

    robot_move(cord_x2-10, cord_y2-20, cord_z2, 180, 0, 180-rz2, c)  #  傳送座標
    # rx = -180  z_min = 400
    rPos2 = receive(c) #接收座標
    print('TCP2')
    print(cord_x2,cord_y2)
    time.sleep(2)  # 5秒後下降
    robot_move(cord_x2, cord_y2, 250, 180, 0, 180-rz2, c)  #  座標為夾取位置
    rPos3 = receive(c)
    return rPos2

def approch_cube(T, T2): # 控制夾爪移動至方塊準備夾取 (寫在主程式內)
    
    rPos, cord_x, cord_y, cord_z = move_1(T)
    rPos2, cord_x2, cord_y2 = move_2(rPos, T2)  # 這裡確保 rPos 的值被傳遞

    TCP_IP = '192.168.0.1' #  Robot IP address. Start the TCP server from the robot before starting this code
    TCP_PORT = 3000  #  Robot Port
    BUFFER_SIZE = 1024  #  Buffer size of the channel, probably 1024 or 4096
    global c
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #  Initialize the communication the robot through TCP as a client, the robot is the server. 
                                                        #  Connect the ethernet cable to the robot electric box first
    c.connect((TCP_IP, TCP_PORT)) 

    robot_move(cord_x2, cord_y2, 180, -180, 0, -145, c)  #  傳送座標
    # rx = -180  z_min = 400
    rPos = receive(c) #接收座標
    print(rPos)

def grip(cmd): #夾取方塊
    ser = serial.Serial('COM5', 9600, timeout = 1)
    if cmd == 1:
        val = ser.write("1".encode('utf-8')) # gripper close
        time.sleep(1)
    elif cmd == 0:
        val = ser.write('0'.encode('utf-8')) # gripper open
        time.sleep(1)

def stackMove(nstack, c): #移動夾取後的方塊至指定位置(no need)
    robot_move(350, -30, 185, 180, 0, 27, c)
    rPos = receive(c)
    robot_move(350, -30, 35+(nstack*50), 180, 0, 27, c)
    rPos = receive(c)
    grip(0)  
    robot_move(350, -30, 185, 180, 0, 27, c)
    rPos = receive(c)
    robot_move(350, 150, 185, 180, 0, 27, c)
    rPos = receive(c)

# pnp移動1  
#sur_cam_pict()

# sur_img = cv2.imread('sur0.jpg')
# sur_fliter(sur_img)
# sur_countour(sur_img)
# sur_color_loc, corners = sur_countour(sur_img)
# sur_points_specs(sur_color_loc, corners)
# sur_cam_pnp()
# T = sur_cam_pnp()
# move_1(T)

# # pnp移動2
# #robot_cam_pict()
# bot_img = cv2.imread('bot0.jpg')
# bot_fliter(bot_img)
# bot_contour(bot_img)
# bot_color_loc, bot_corners = bot_contour(bot_img)
# bot_point_specs (bot_color_loc,bot_corners)
# T2 = bot_pnp()
# cord_x, cord_y, cord_z = move_1(T)
# move_2(T2)

#夾取方塊移動至指定點
# approch_cube()
# gripMove(1) 
# stackMove(1,c)