import cv2
import matplotlib.pyplot as plt
import numpy


image = cv2.imread('mii.jpeg')

image_tr = numpy.transpose(image)
image_move = numpy.moveaxis(image,2,0)
image_reshape = numpy.reshape(image,(3,465,828))

output_file = "output_briness.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = 15
frame = 10

out = cv2.VideoWriter(output_file, fourcc, fps, (image.shape[1], image.shape[0]))

a = 1
b = 0
y = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0] 

def briness(img,a,b,y):                                         #เป็นฟังค์ชันปรับความสว่างตามสมการที่กำหมนค่าใว้ตามโจทย์
    H,W,C = img.shape                                           #จะเข้า Loop เปลี่ยนค่าที่ละ Pixcel
    Rev = numpy.zeros_like(img)            
    for c in range(0,C):
        for w in range(0,W):
            for h in range(0,H):
                cal = a * (img[h][w][c]**y) + b                 #สมการการคำนวน
                if cal <= 255:                                  #ดักค่าใว้ไม่เกิน 255 
                    Rev[h][w][c] = cal                              
                else:
                    Rev[h][w][c] = 255
    return Rev


for i in range(20):
    brinessOutput = briness(image,a,b,y[i])                     #สร้าง Video ความยาวภาพละ 100 เฟรม
    for f in range(frame):
        out.write(brinessOutput)