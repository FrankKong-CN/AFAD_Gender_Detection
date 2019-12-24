from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils

from model import *

BATCH_SIZE = 128
EPOCH = 100

CLASSES = 2

OPTIMIZER = Adam()

# 加载数据
X = np.load("data/data_X.npy")
y = np.load("data/data_y.npy")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# 对y进行one-hot编码
Y_train = np_utils.to_categorical(y_train, CLASSES)
Y_test = np_utils.to_categorical(y_test, CLASSES)
X_train = X_train / 255
X_test = X_test / 255

model = get_model()

model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCH)

model.save("model_v1.0.0/AFAD_Gender_Detection_2019_12_6.h5")
