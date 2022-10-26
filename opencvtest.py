
import cv2
import numpy as np
import matplotlib.pyplot as plt

#载入原图，并转为灰度图像
img_original=cv2.imread('C:/Users/luke/Desktop/888.jpg')
img_gray=cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(img_gray,(11,11),0)
binaryIMG = cv2.Canny(blurred,20,160)

#求二值图像
retv,thresh=cv2.threshold(binaryIMG,125,255,1)
#寻找轮廓
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
 #CV2.moments會傳回一系列的moments值，我們只要知道中點X, Y的取得方式是如下進行即可。
   M = cv2.moments(c)
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
# 在中心點畫上黃色實心圓
cv2.circle(img_original, (cX, cY), 10, (1, 227, 254), -1)


#绘制轮廓
cv2.drawContours(img_original,contours,-1,(0,0,255),3,lineType=cv2.LINE_AA)
#显示图像
cv2.imshow('Contours',img_original)
cv2.waitKey()
cv2.destroyAllWindows()
print(hierarchy)