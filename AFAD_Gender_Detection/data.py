"""加载各年龄段男女图片，保存至numpy文件夹中"""

import numpy as np
import imageio
from glob import glob
import matplotlib.pyplot as plt


def load_data():
    """首先加载男性图片"""
    data = np.empty((0, 48, 48, 3))
    num_male = 0
    num_female = 0
    for age in range(15, 41):
        print("load age is %d years old male..." % age)
        img_path_0 = glob("D://dataset/AFAD-Full/AFAD-train/" + str(age) + "/0/*.jpg")
        num_0 = len(img_path_0)
        num_male += num_0
        for i, file in enumerate(img_path_0):
            img = imageio.imread(file, as_gray=False)
            img = np.array(img, dtype='float32')
            data = np.append(data, img)
            data = data.reshape((-1, 48, 48, 3))

    '''加载女性图片'''
    for age in range(15, 41):
        print("load age is %d years old female..." % age)
        img_path_1 = glob("D:/dataset/AFAD-Full/AFAD-train/" + str(age) + "/1/*.jpg")
        num_1 = len(img_path_1)
        num_female += num_1
        for i, file in enumerate(img_path_1):
            img = imageio.imread(file, as_gray=False)
            img = np.array(img, dtype='float32')
            data = np.append(data, img)
            data = data.reshape((-1, 48, 48, 3))

    '''加载标签'''
    y_0 = np.zeros((1, num_male))
    y_1 = np.ones((1, num_female))
    y = np.append(y_0, y_1)

    return data, y


if __name__ == '__main__':

    X = np.load("data/data_X.npy")
    y = np.load("data/data_y.npy")

    plt.imshow(X[0] / 255)
    plt.show()
