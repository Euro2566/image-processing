import cv2 as cv
import numpy as np

img_1 = cv.imread("mii2.jpeg")
img2 = cv.imread("mii.jpeg")
resize_img1 = cv.resize(img_1, (200, 200))
resize_img2 = cv.resize(img2, (200, 200))

output_file = "output_file.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')

fps = 60
frame = 20
trans_duration = 30

out = cv.VideoWriter(output_file, fourcc, fps, (resize_img1.shape[1], resize_img1.shape[0]))

w = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

# สร้างวิดีโอที่ภาพเปลี่ยนแปลงตามค่า weight ของภาพทั้งสอง
for w1, w2 in zip(w, w[::-1]):
    image_result = cv.addWeighted(resize_img1, w1, resize_img2, w2, 0)
    print(w1,w2)
    for i in range(frame):
        out.write(image_result)

# สร้างวิดีโอที่ภาพเปลี่ยนแปลงตามค่า weight ของภาพทั้งสองใหม่อีกครั้ง
for w2, w1 in zip(w, w[::-1]):
    image_result = cv.addWeighted(resize_img1, w1, resize_img2, w2, 0)
    for i in range(frame):
        out.write(image_result)

out.release()