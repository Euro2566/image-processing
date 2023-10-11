import cv2
import matplotlib.pyplot as plt
import numpy


image = cv2.imread('mii.jpeg')

image_tr = numpy.transpose(image)
image_move = numpy.moveaxis(image,2,0)
image_reshape = numpy.reshape(image,(3,465,828))
def Bittv(img):
    H,W,C = img.shape
    for c in range(0,C):
        for w in range(0,W):
            for h in range(0,H):
                img[h][w][c] = (img[h][w][c]/255)*128
    return img

#A = Bittv(image)

#print(A)
print(image_tr.shape)
print(image_move.shape)

plt.figure(figsize=(15, 10))



plt.subplot(2, 4, 1)
plt.imshow(image)
plt.title('Original')

plt.subplot(2, 4, 2)
plt.imshow(Bittv(image))
plt.title('down')

plt.show()