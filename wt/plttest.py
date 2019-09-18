import matplotlib.pyplot as plt
import cv2


image1=plt.imread('image/lenna.png')
plt.ion()  # 打开交互模式
# 同时打开两个窗口显示图片
plt.figure()  # 图片一
plt.imshow('image1',image1)

plt.figure()  # 图片二
plt.imshow('image2','image/lenna.png')
# 显示前关掉交互模式
plt.ioff()
plt.show()
