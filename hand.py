from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


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
    show_img("edged", edged)

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
    show_img("Image", origin)
    return origin

if __name__ == "__main__":
    image_path = 'C:/Users/luke/Desktop/999.png'
    image = cv_imread(image_path)
    gray = preprocessing(image)
    contours2, pixelsPerMetric = edge_detect(gray)
    measure(image, contours2, pixelsPerMetric)