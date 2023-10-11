import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread                         
from skimage.exposure import cumulative_distribution 

# อ่านภาพจากไฟล์
image_1 = cv2.imread("mii.jpeg")
image_2 = cv2.imread("T1.jpg")

# แยกสีของภาพออกเป็นช่อง RGB
im1b, im1g, im1r = cv2.split(image_1)
im2b, im2g, im2r = cv2.split(image_2)

# สร้าง array สำหรับค่าพิกเซล
pixels = np.arange(256)

# สร้างฟังก์ชันสำหรับคำนวณ Cumulative Distribution Function (CDF)
def getCDF(image):
    #ฟังก์ชันนี้รับภาพสีเป็นอินพุท (เช่น ภาพช่อง R, G, หรือ B) 
    #และทำการคำนวณ Cumulative Distribution Function (CDF) ของค่าสีในภาพนั้น
    cdf, bins = cumulative_distribution(image)
    cdf = np.insert(cdf, 0, [0]*bins[0])
    #การคำนวณ CDF นั้นจะเริ่มจากการคำนวณ histogram ของค่าสีในภาพ 
    #และแปลง histogram เป็น CDF โดยทำการสะสมความถี่ของค่าสีที่น้อยกว่าหรือเท่ากับค่านั้น

    #CDF ที่ได้จะถูกปรับให้มีค่าเริ่มต้นที่ 0 และสิ้นสุดที่ 1 เพื่อให้เป็นค่าที่สามารถนำไปใช้ในกระบวนการปรับค่าสีได้
    cdf = np.append(cdf, [1]*(255-bins[-1]))
    return cdf

# สร้างฟังก์ชันสำหรับการจับคู่ Histogram
def histMatch(cdfInput, cdfTemplate, imageInput):
    #ฟังก์ชันนี้รับ CDF ของภาพอินพุท (cdfInput), CDF ของภาพเทมเพลต (cdfTemplate), 
    #และภาพอินพุทที่ต้องการปรับปรุงค่าสี (imageInput)

    #การทำงานของฟังก์ชันนี้คือการทำการจับคู่ค่าสีของภาพอินพุท (imageInput) 
    #โดยใช้ CDF ของภาพเทมเพลต (cdfTemplate) เพื่อปรับปรุงแบบกระจายค่าสี

    #ขั้นตอนการทำงานคือ การใช้ฟังก์ชัน np.interp เพื่อแปลงค่า CDF ของภาพอินพุท (cdfInput) 
    #ให้อยู่ในรูปแบบของ CDF ของภาพเทมเพลต (cdfTemplate)

    #จากนั้นค่าสีในภาพอินพุทจะถูกแทนที่ด้วยค่าสีที่ได้จากการจับคู่ดังกล่าว 
    #ผลลัพธ์ที่ได้คือภาพที่ค่าสีถูกปรับแบบสอดคล้องกับค่าสีของภาพเทมเพลต
    pixelValues = np.arange(256)
    new_pixels = np.interp(cdfInput, cdfTemplate, pixels)
    imageMatch = (np.reshape(new_pixels[imageInput.ravel()], imageInput.shape)).astype(np.uint8)
    return imageMatch

# แสดงรูปภาพเริ่มต้น
plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
plt.title("Input")
ax1 = plt.subplot(3, 3, 2)
ax2 = plt.subplot(3, 3, 3)

# คำนวณและแสดง Histogram และ CDF สีแต่ละช่องของภาพที่ 1 (Input)
for i, c in enumerate('bgr'):
    hist = cv2.calcHist([image_1], [i], None, [256], [0, 256])
    cdf1 = np.cumsum(hist) / sum(hist)
    ax1.plot(hist, c)
    ax2.plot(cdf1, c)
    ax2.set_ylabel("CDF")

# แสดงรูปภาพเปรียบเทียบ (Template)
plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
plt.title("Template")
ax3 = plt.subplot(3, 3, 5)
ax4 = plt.subplot(3, 3, 6)

# คำนวณและแสดง Histogram และ CDF สีแต่ละช่องของภาพที่ 2 (Template)
for i, c in enumerate('bgr'):
    hist = cv2.calcHist([image_2], [i], None, [256], [0, 256])
    cdf2 = np.cumsum(hist) / sum(hist)
    ax3.plot(hist, c)
    ax4.plot(cdf2, c)
    ax4.set_ylabel("CDF")

# สร้างภาพที่ผ่านกระบวนการ Histogram Matching
image_result = np.zeros((image_1.shape)).astype(np.uint8)

for i in range(3):
    cdfInput = getCDF(image_1[:,:,i])
    cdfTemplate = getCDF(image_2[:,:,i])
    image_result[:,:,i] = histMatch(cdfInput, cdfTemplate, image_1[:,:,i])

# แสดงภาพที่ผ่านกระบวนการ Matching
plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
plt.title("Matching")
ax5 = plt.subplot(3, 3, 8)
ax6 = plt.subplot(3, 3, 9)

# คำนวณและแสดง Histogram และ CDF สีแต่ละช่องของภาพที่ผ่านกระบวนการ Matching
for i, c in enumerate('bgr'):
    hist = cv2.calcHist([image_result], [i], None, [256], [0, 256])
    cdf2 = np.cumsum(hist) / sum(hist)
    ax5.plot(hist, c)
    ax6.plot(cdf2, c)
    ax6.set_ylabel("CDF")

# บันทึกภาพที่ผ่านกระบวนการ Matching เป็นไฟล์
cv2.imwrite('h:\image_processing/image_result.jpg', image_result)

# แสดงผลกราฟและรูปภาพ
plt.tight_layout()
plt.show()