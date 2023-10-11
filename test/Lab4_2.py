#Array, image processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
#Model Operation
from keras import Model, Input
import keras.utils as image
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam,SGD,RMSprop,Adadelta
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scipy import signal

# io
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#ทำโครงข่ายของการประมาลผล หรือ การทำโมเดล Connect Encoder and Decoder Model
def create_autoencoder(optimizer= 'Adam',learning_rates = None):
    Input_img = Input(shape=(80,80,3))
    otp = None

    x1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(Input_img)
    x2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x1)
    x2 = MaxPool2D((2,2))(x2)
    x3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x2)
    encoded = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x3)

    x4 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(encoded)
    x5 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x4)
    x5 = UpSampling2D((2,2))(x5)
    x6 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x5)
    x7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x6)
    decoded = Conv2D(3, (3,3), padding = 'same')(x7)

    autoencoder = Model(Input_img, decoded)
    if optimizer == 'Adam':
        otp = Adam(learning_rates)
    elif optimizer == 'SGD':
        otp = SGD(learning_rates)
    elif optimizer == 'RMSprop':
        otp = RMSprop(learning_rates)
    elif optimizer == 'Adadelta':
        otp = Adadelta(learning_rates)

    autoencoder.compile(optimizer=otp, loss = 'mean_squared_error', metrics = ['mean_squared_error'])

    return autoencoder

intensity = [0, 1]
k = 42
scalar = 0.8
noise_mean = 0.3 
noise_std = scalar
noise_factor = scalar
ImgArray = []

#ส่วนของการอ่านภาพมาจากโฟเคอร์ที่เก็บภาพแล้วใส่ใน array ที่เราสร้างไว้
imgs = glob.glob('face_mini/**/*.jpg')
for img in imgs:
    loadImg = cv2.imread(img)
    ResizeImg = cv2.resize(loadImg, (80, 80))
    rgb_image = cv2.cvtColor(ResizeImg, cv2.COLOR_BGR2RGB)
    ImgArray.append(rgb_image)

#ส่วนของการทำ Nomirise
ImgsArray = np.array(ImgArray)
ImgsArray = ImgsArray / 255

train_x, test_x = train_test_split(ImgsArray, random_state=k, test_size=0.3)

train_x, val_x = train_test_split(train_x, random_state=k, test_size=0.2)


train_x_noise = train_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=train_x.shape))
val_x_noise = val_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=val_x.shape))
test_x_noise = test_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=test_x.shape))


epoch = [2,4,8,16]
batch_size = [16,32,64,128]
optimizer = ['SGD', 'RMSprop', 'Adadelta', 'Adam']
learning_rates = [0.01, 0.001, 0.0001, 0.00001] 


#ใช้สร้าง Model โดยมาจากฟังค์ชั่นที่เราสร้าง
autoencoder = create_autoencoder(optimizer[3],learning_rates[2])
callback = EarlyStopping(monitor='loss', patience=3)
#ส่วนของการเทรนโมเดลโดยที่เราใส่ภาพที่ไม่มี noise และภาพที่มี noise เพื่อจะได้ภาพที่ถูกลบ noise ออกเป็น output
history = autoencoder.fit(train_x_noise, train_x,epochs=epoch[2],batch_size=batch_size[0],shuffle=True,validation_data=(val_x_noise, val_x),callbacks=[callback],verbose=1)

#ส่งข้อมูลเข้าไปทำนาย
predictions_test = autoencoder.predict(test_x_noise)

#้ส่วนการแสดงผลกราฟ
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


#ส่วนการแสดงผลภาพที่ยังไม่ใส่ noise 
plt.figure(figsize=(12, 12))
for i in range(0,10):
    plt.subplot(3, 10, i+1)
    plt.imshow(test_x[i])

#ส่วนการแสดงผลภาพที่ยังไม่ใส่ noise แล้ว
for i in range(0,10):
    plt.subplot(3, 10, 11+i)
    plt.imshow(test_x_noise[i])

#ส่วนการแสดงผลภาพที่ลบ noise ออกไป 
for i in range(0,10):
    plt.subplot(3, 10, 21+i)
    plt.imshow(predictions_test[i])


plt.show()