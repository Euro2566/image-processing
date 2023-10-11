#Array, image processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
#Model Operation
from keras import Model, Input
import keras.utils as image
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scipy import signal

# io
import glob
from tqdm import tqdm
import warnings;
warnings.filterwarnings('ignore')

intensity = [0, 1]
k = 42
scalar = 0.8
noise_mean = 0.3 
noise_std = scalar
noise_factor = scalar
ImgArray = []


# โหลดภาพและปรับขนาดเป็น 80x80 pixels
imgs = glob.glob('face_mini/**/*.jpg')
for img in imgs:
    loadImg = cv2.imread(img)
    ResizeImg = cv2.resize(loadImg, (80, 80))
    rgb_image = cv2.cvtColor(ResizeImg, cv2.COLOR_BGR2RGB)
    ImgArray.append(rgb_image)

# แปลงรายการภาพเป็น NumPy array และทำการ normalize ค่า pixel เข้าในช่วง 0-1
ImgsArray = np.array(ImgArray)
ImgsArray = ImgsArray / 255

# แบ่งชุดฝึกสอน, ชุดทดสอบ, และชุด validation
train_x, test_x = train_test_split(ImgsArray, random_state=k, test_size=0.3)
train_x, val_x = train_test_split(train_x, random_state=k, test_size=0.2)

# เพิ่ม noise ลงในภาพ
train_x_noise = train_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=train_x.shape))
val_x_noise = val_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=val_x.shape))
test_x_noise = test_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=test_x.shape))

# แสดงภาพที่มีและไม่มี noise
plt.figure(figsize=(12, 12))
for i in range(0,10):
    plt.subplot(2, 10, i+1)
    plt.imshow(test_x[i])

for i in range(0,10):
    plt.subplot(2, 10, 11+i)
    plt.imshow(test_x_noise[i])


plt.tight_layout()
plt.show()
