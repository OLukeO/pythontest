import cv2
import mediapipe as mp
import os
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
from flask import Flask,send_file

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

def bodyandmouthcheck(): ###身體和嘴範圍
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

                if x*640 > 130 and x*640 < 600 and y*480 > 60 and y*480 < 450 :
                    mp4 = send_file("C:/Users/luke/Desktop/Sequence 02.mp4")
                    mp4.headers['tok'] = "123456"
                    return mp4
                    break

            except:
                pass
            cv2.rectangle(image, (190, 60), (540, 185), (0, 0, 255), 2)  # 畫出觸碰區
            cv2.rectangle(image, (130, 180), (600,450), (0, 0, 255), 2)  # 畫出觸碰區
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def hand(): ###手掌偵測
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        run = True  # 設定是否更動觸碰區位置
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            img = cv2.resize(img, (540, 320))  # 調整畫面尺寸
            size = img.shape  # 取得攝影機影像尺寸
            w = size[1]  # 取得畫面寬度
            h = size[0]  # 取得畫面高度

            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換成 RGB
            results = hands.process(img2)  # 偵測手掌
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x = hand_landmarks.landmark[3].x * w  # 取得食指末端 x 座標
                    y = hand_landmarks.landmark[3].y * h  # 取得食指末端 y 座標
                    x1 = hand_landmarks.landmark[9].x * w  # 取得中指末端 x 座標
                    y1 = hand_landmarks.landmark[9].y * h  # 取得中指末端 y 座標
                    x2 = hand_landmarks.landmark[17].x * w  # 取得中指末端 x 座標
                    y2 = hand_landmarks.landmark[17].y * h  # 取得中指末端 y 座標
                    x3 = hand_landmarks.landmark[0].x * w  # 取得中指末端 x 座標
                    y3 = hand_landmarks.landmark[0].y * h  # 取得中指末端 y 座標
                    # print(x,y)
                    if x > 150 and x < 420 and y > 180 and y < 300 and x1 > 150 and x1 < 420\
                            and y1 > 180 and y1 < 300 and x2 > 150 and x2 < 420\
                            and y2 > 180 and y2 < 300 and x3 > 150 and x3 < 420\
                            and y3 > 180 and y3 < 300:
                            x = int(x)
                            x1 = int(x1)
                            x2 = int(x2)
                            x3 = int(x3)
                            y = int(y)
                            y1 = int(y1)
                            y2 = int(y2)
                            y3 = int(y3)
                            cv2.imwrite("pill.jpg",img2[min(y,y1,y2,y3):max(y,y1,y2,y3), min(x,x1,x2,x3):max(x,x1,x2,x3)])
                            image_path = 'C:/Users/luke/PycharmProjects/pythonProject/pill.jpg'
                            image = cv_imread(image_path)
                            gray = preprocessing(image)
                            contours2, pixelsPerMetric = edge_detect(gray)
                            measure(image, contours2, pixelsPerMetric)
                            print("yes")
                    # 將節點和骨架繪製到影像中
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style())

            cv2.rectangle(img, (150, 180), (420,300), (0, 0, 255), 2)  # 畫出觸碰區
            cv2.imshow('oxxostudio', img)
    cap.release()
    cv2.destroyAllWindows()
#######三
# 讀取中文路徑圖檔(圖片讀取為BGR)
def cv_imread(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

# 顯示圖檔
def show_img(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)

# 圖片預處理
def preprocessing(image):
    # 灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯濾波
    gaussian = cv2.GaussianBlur(gray, (9, 9), 0)

    # 開運算去除白色噪點
    kernel = np.ones((9, 9), np.uint8)
    open = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN,
                            kernel, iterations=3)
    return open

def edge_detect(image):
    # 以canny邊緣檢測算法獲取目標(50~100為閾值)
    edged = cv2.Canny(image, 50, 100) #低於50刪除 高於100留下
    #show_img("edged", edged)

    # 在邊緣圖像中尋找物體輪廓
    contours1, hierarchy = cv2.findContours(edged.copy(),
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    # 物體輪廓由左到右進行排序
    (contours2, _) = contours.sort_contours(contours1)
    # 若輪廓面積小於100，視為噪音濾除
    contours2 = [i for i in contours2 if cv2.contourArea(i) > 40]

    # 初始化 pixels per metric
    pixelsPerMetric = None

    return contours2, pixelsPerMetric

# 畫切線
def measure(image, contours2, pixelsPerMetric):
    j = 0
    origin = image.copy()
    for i in contours2:
        # 計算出物品輪廓之外切線框
        box = cv2.minAreaRect(i)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() \
              else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # 左上角座標開始順時針排序，並畫出外切線框
        box = perspective.order_points(box)
        cv2.drawContours(origin, [box.astype("int")], -1, (0, 255, 0), 2)

        # 畫書外切線框端點
        for (x, y) in box:
            cv2.circle(origin, (int(x), int(y)), 5, (0, 0, 255), -1)
            j += 1
    j /= 4
    if j > 0:
        print("yes")
        print(j)
    elif j <= 0:
        print("no")
    #show_img("Image", origin)
    #return origin




@app.route("/")
def hello():
    return bodyandmouthcheck()

if __name__ == "__main__":
    app.run(host="0.0.0.0" ,port=5000)