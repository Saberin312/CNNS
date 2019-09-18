import numpy as np
import pywt
import cv2
import os
from time import sleep
from threading import Thread
import matplotlib.pyplot as plt

os.chdir('F:/download/pixiv')
img = cv2.imread("60973718_p0.png")
cv2.imshow('img', img)
cv2.waitKey()
print(img.shape)

img1 = img.copy()
print(img1.shape)
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
cv2.imshow('img1', img1)
cv2.waitKey()


def close(time):
    sleep(time)
    plt.close()


plt.figure('图片小波变换')
coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

# 将各个子图进行拼接，最后得到一张图
AH = np.concatenate([cA, cH], axis=1)
VD = np.concatenate([cV, cD], axis=1)
img = np.concatenate([AH, VD], axis=0)
# thread1=Thread(target=close,args=(5,))
# thread1.start()

plt.ion()
plt.show()
print("执行完毕！")
# plt.show()

# plt.close('all')
# wtimg1=pywt.dwt2(img1)
# lenna = cv2.imread("image/lenna.png")
# lenna = cv2.imshow('lenna', lenna)
# cv2.waitKey(0)
# plt.show(img)
# cv2.imshow(img)
# img = cv2.resize(img, (500, 500))
# 多通道变成单通道
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
#
# plt.figure('二维小波一级变换')
# coeffs = pywt.dwt2(img, 'haar')
# cA, (cH, cV, cD) = coeffs
#
# # 将各个子图进行拼接，最后得到一张图
# AH = np.concatenate([cA, cH], axis=1)
# VD = np.concatenate([cV, cD], axis=1)
# img = np.concatenate([AH, VD], axis=0)
#
# # 显示为灰度图
# plt.imshow(img, 'gray')
# plt.title('result')
# plt.show()

# plt.imshow(img,'gray')
# plt.show()
