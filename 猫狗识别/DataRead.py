# coding:utf-8

import numpy as np
import os
from PIL import Image


def get_data(path, height, width, alpha):
    alpha = 1 - alpha
    cats = []
    dogs = []
    cat_labels = []
    dog_labels = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    names = os.listdir(path)
    for name in names:
        li = name.split(".")
        if "cat" == li[0]:
            cat_path = os.path.join(path, name)
            # 剪裁每幅图像的大小
            cat = Image.open(cat_path).resize((height, width))
            cat = np.array(cat)
            # 将图像的大小缩减到0-1范围内
            cat = cat * (1/255)
            cats.append(cat)
            label = np.array([1.0, 0.0])
            cat_labels.append(label)
        else:
            dog_path = os.path.join(path, name)
            dog = Image.open(dog_path).resize((height, width))
            dog = np.array(dog)
            dog = dog * (1/255)
            dogs.append(dog)
            label = np.array([0.0, 1.0])
            dog_labels.append(label)

    # 划分训练数据和测试数据
    train_num = int(alpha * len(cats))
    x_train.extend(cats[:train_num])
    x_train.extend(dogs[:train_num])
    y_train.extend(cat_labels[:train_num])
    y_train.extend(dog_labels[:train_num])
    x_test.extend(cats[train_num:])
    x_test.extend(dogs[train_num:])
    y_test.extend(cat_labels[train_num:])
    y_test.extend(dog_labels[train_num:])

    # 打乱训练数据
    np.random.seed(2)
    np.random.shuffle(x_train)
    np.random.seed(2)
    np.random.shuffle(y_train)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


# path = r"E:\python\PythonSpace\Git\TensorFlow\猫狗识别\data\train"
# height = 128
# width = 128
# alpha = 0.2
# x_train, x_test, y_train, y_test = get_data(path, height, width, alpha)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# print(y_train[0])
# print(x_train[0])