
import numpy as np
import cv2
import matplotlib.pyplot as plt

img_1 = cv2.imread("mii2.jpeg")
img2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

# ปรับขนาดภาพเป็น 1000x1000
resize_img2 = cv2.resize(img2, (1000, 1000))

# แสดงภาพต้นฉบับ
plt.subplot(1, 3, 1)
plt.imshow(resize_img2)
plt.title('Original')

height, width,array = resize_img2.shape

# สร้างแมสก์ขนาดเดียวกับภาพที่อยู่ในกรอบสี่เหลี่ยม
mask = np.zeros((height, width,array), dtype=np.uint8)
cv2.rectangle(mask, (300, 200), (650, 600), (255,255,255), -1)

# แสดงแมสก์
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Image Mask')

# ประมวลผลภาพด้วยการกำหนดแมสก์
masked_image = cv2.bitwise_and(resize_img2,mask)

# แสดงผลลัพธ์ภาพที่ประมวลผลแล้ว
plt.subplot(1, 3, 3)
plt.imshow(masked_image)
plt.title('Bitwise_AND() result')

# ปรับระยะห่างให้กับ subplot
plt.subplots_adjust(wspace=0.5)

# แสดงกราฟทั้งหมด
plt.show()