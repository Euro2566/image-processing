import cv2
import matplotlib.pyplot as plt
import numpy


image = cv2.imread('mii2.jpeg')

image_tr = numpy.transpose(image)
image_move = numpy.moveaxis(image,2,0)
image_reshape = numpy.reshape(image,(3,465,828))

print(image.shape)
print(image_tr.shape)
print(image_move.shape)

plt.figure(figsize=(15, 10))



plt.subplot(2, 4, 1)
plt.imshow(image[:,:,0],cmap='gray')
plt.title('Original')


plt.subplot(2, 4, 2)
plt.imshow(image_tr[0,:,:],cmap='gray')
plt.title('Transpose')


plt.subplot(2, 4, 3)
plt.imshow(image_move[0,:,:],cmap='gray')
plt.title('Moveaxis')

plt.subplot(2, 4, 4)
plt.imshow(image_reshape[0,:,:],cmap='gray')
plt.title('Reshape')





plt.show()