from keras.models import load_model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

model = load_model("model_v1.0.0/AFAD_Gender_Detection_2019_12_6.h5")

CLASSES = 2
BATCH_SIZE = 128

# 加载数据
X = np.load("data/data_X.npy")
y = np.load("data/data_y.npy")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# 对y进行one-hot编码
Y_test = np_utils.to_categorical(y_test, CLASSES)
X_test = X_test / 255

score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)

print(score)
