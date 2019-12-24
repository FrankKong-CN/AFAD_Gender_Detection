from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

# 48 * 48 的三通道
IMG_CHANNELS = 3
IMG_ROWS = 48
IMG_COLS = 48

# 常量
CLASSES = 2


def get_model():
    # 创建模型
    model = Sequential()

    # 搭建模型
    # conv1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    # conv2
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    # conv3
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    # 全连接层
    model.add(Flatten())
    model.add(Dense(520, activation='relu'))
    # 输出
    model.add(Dense(CLASSES, activation='softmax'))

    return model


if __name__ == '__main__':
    model = get_model()
    model.summary()
    plot_model(model, to_file='model_v1.0.0/model.png', show_shapes=True)
