import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("‪F:\download\pixiv\60973718_p0.png")
# img = cv2.resize(img, (500, 500))
# 多通道变成单通道
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

plt.figure('二维小波一级变换')
coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

# 将各个子图进行拼接，最后得到一张图
AH = np.concatenate([cA, cH], axis=1)
VD = np.concatenate([cV, cD], axis=1)
img = np.concatenate([AH, VD], axis=0)

# 显示为灰度图
plt.imshow(img, 'gray')
plt.title('result')
plt.show()

# plt.imshow(img,'gray')
# plt.show()
