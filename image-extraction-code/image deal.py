import cv2
import numpy as np
import random
import mahotas as mt


if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\dell\Desktop\tumor.jpg", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (13, 13), 0)  # 高斯滤波 可以进行图像平滑处理 降噪
    equ_img = cv2.equalizeHist(blur)  # 图像增强，得到直方图均衡化后的图像
    cv2.imshow('', equ_img)
    cv2.waitKey(0)
