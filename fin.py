import cv2
import os
import mediapipe as mp
import numpy as np

#mediapipe的手和身體關鍵點模組匯入
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#皮膚顏色上下限
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)


cap = cv2.VideoCapture(0)

while True:
    #ret, frame = cap.read()
    frame = cv2.imread('C:/Users/luke/Desktop/77777.jpg')
    cv2.rectangle(frame, (40, 80), (100, 140), (0, 0, 255), 3)

    imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    skinYCrCb = cv2.bitwise_and(frame, frame, mask=skinRegionYCrCb)
    frame = np.hstack([frame, skinYCrCb])


    #######
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    binaryIMG = cv2.Canny(blurred, 20, 160)
    # 求二值图像
    retv, thresh = cv2.threshold(binaryIMG, 125, 255, 1)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    a = cv2.drawContours(frame, contours, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    if a.all:
        print("yes")
    else:
        print("no")
    ########


    frame = cv2.resize(frame,(540,300))  # 縮小尺寸，加快速度
    keyName = cv2.waitKey(1)
    cv2.imshow('oxxostudio', frame)

cap.release()
cv2.destroyAllWindows()
