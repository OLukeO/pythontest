import cv2
import numpy as np
import mediapipe as mp
import multiprocessing as mps
from imutils import perspective
from imutils import contours
import imutils


class YuuzuCam:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.parent_conn, child_conn = mps.Pipe()
        self.p = mps.Process(target=self.update, args=(child_conn, url))
        self.p.daemon = True
        self.p.start()

    def update(self, conn, url):
        print("Cam Loading...")
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        print("Cam Loaded...")
        run = True

        while run:
            # grab frames from the buffer
            self.cap.grab()
            # receive input data
            rec_dat = conn.recv()

            if rec_dat == 1:
                # if frame requested
                ret, frame = self.cap.read()
                conn.send(frame)

            elif rec_dat == 2:
                # if close requested
                self.cap.release()
                run = False

        print("Connection Closed")
        conn.close()

    def end(self):
        # send closure request to process
        self.parent_conn.send(2)

    def get_frame(self, resize=None):
        self.parent_conn.send(1)
        frame = self.parent_conn.recv()

        # reset request
        self.parent_conn.send(0)

        # resize if needed
        if resize is None:
            return frame
        else:
            return self.rescale_frame(frame, resize)

    def rescale_frame(self, frame, percent):
        if frame is None:
            self.end()
            return None

        return cv2.resize(frame, (0, 0), fx=percent, fy=percent)


cam = YuuzuCam('rtsp://root:awedvhu0808@120.110.115.130:554/axis-media/media.amp')

if not cam.p.is_alive():
    print("Camera is not alive")

mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
#############################
def bodyandmouthcheck(): ###身體和嘴範圍
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cam.p.is_alive():
            ret, frame = cam.get_frame()

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
                    print("yes")

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

        #cap.release()
        cv2.destroyAllWindows()

def hand(): ###手掌偵測
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        if not cam.p.is_alive():
            print("Cannot open camera")
            exit()

        run = True  # 設定是否更動觸碰區位置
        while True:
            ret, img = cam.get_frame()
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
    #cap.release()
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
#############################
def calculate_distance(a, b):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    distance = np.linalg.norm(a - b)

    return distance


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angles = np.abs(radians * 180.0 / np.pi)

    if angles > 180.0:
        angles = 360 - angles

    return angles


def main():
    bodyandmouthcheck()
    hand()

    # Curl counter variables
    counter = 0
    trigger = False
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while 1:
            frame = cam.get_frame()

            # Get Size of frame
            width = 1920
            height = 1080

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
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x * width,
                         landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y * height]
                index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * width,
                         landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * height]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                distances = calculate_distance(index, mouth)

                # Visualize angle
                cv2.putText(image, str(distances),
                            tuple(np.multiply(index, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                # if angle > 160:
                #     stage = "down"
                # if angle < 30 and stage == 'down':
                #     stage = "up"
                #     counter += 1
                #     print(counter)
                if distances <= 20:
                    if not trigger:
                        trigger = True
                        counter += 1
                elif distances >= 60:
                    if trigger:
                        trigger = False

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow("frame", image)

            key = cv2.waitKey(1)
            if key == 13:  # 13 is the Enter Key
                break

    cv2.destroyAllWindows()
    cam.end()

if __name__ == '__main__':
    main()
    mps.freeze_support()