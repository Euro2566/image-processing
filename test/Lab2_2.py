import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("mii2.jpeg")


b, g, r = cv2.split(image)                                          #แยกสีของแต่ละช่อง


equalized_r = cv2.equalizeHist(r)                                   #ทำ equalize ของแต่ละสี
equalized_g = cv2.equalizeHist(g)
equalized_b = cv2.equalizeHist(b)


equa_colored = cv2.merge((equalized_b, equalized_g, equalized_r))   #รวมสีเป็นภาพจากการทำ equalize ของแต่ละสี


plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))                  #แสดงภาพ Original
plt.title("Original")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(equa_colored, cv2.COLOR_BGR2RGB))           #แสดงภาพที่ทำ equalize
plt.title("Equalized")

plt.subplot(2, 2, 2)
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])        #สร้างกราฟสีของภาพต้นฉบับ
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.title("Original Histogram")

plt.subplot(2, 2, 4)
for i, color in enumerate(colors):
    hist = cv2.calcHist([equa_colored], [i], None, [256], [0, 256]) #สร้างกราฟสีของภาพที่ทำ equalize
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.title("Equlized Histogram")


plt.tight_layout()
plt.show()