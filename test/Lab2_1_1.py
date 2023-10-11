import cv2
import matplotlib.pyplot as plt
import numpy


image = cv2.imread('mii.jpeg')

image_tr = numpy.transpose(image)
image_move = numpy.moveaxis(image,2,0)
image_reshape = numpy.reshape(image,(3,465,828))

output_file = "output_contras.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = 60
frame = 20

out = cv2.VideoWriter(output_file, fourcc, fps, (image.shape[1], image.shape[0]))

a = [0.5,1]
b = [2,4,6,8,10,12,14,16,18,20]
print(image)

def contrasCal(img,a,b):                                 #สร้างความคมชัดของภาพโดยการเข้าสมาการเปลี่ยนทีละ Pixcel
    H,W,C = img.shape
    Rev = numpy.zeros_like(img)                                                                   
    for c in range(0,C):
        for w in range(0,W):
            for h in range(0,H):
                cal = a * img[h][w][c] + b              #สมการคำนวน
                if cal <= 255:                          #ดักค่าไม่เกิน 255
                    Rev[h][w][c] = cal                      
                else:
                    Rev[h][w][c] = 255
    return Rev


for i in range(2):                                      #ปรับค่าตัวคูณ a                   
    for j in range(10):                                 #ปรับค่าบวกของ b 
        contrasOutput = contrasCal(image,a[i],b[j])
        for f in range(frame):                          #สร้างภาพละ 20 เฟรม
            out.write(contrasOutput)
print("------------------")
print(contrasOutput)