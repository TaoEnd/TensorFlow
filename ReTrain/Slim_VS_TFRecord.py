# coding:utf-8

# 使用slim来从头开始构建模型

import tensorflow as tf
import os
import random
import math
import sys

# 测试集数据
NUM_TEST = 500
# 随机种子
RANDOM_SEED = 0
# 数据块，当样本很大时，需要将其划分成多个块
NUM_SHARDS = 5
# 数据集路径
DATASET_DIR = r"E:\python\PythonSpace\Data\slim\picture"
# label文件位置，该文件中是数字和label的对应关系
LABELS_PATH = r"E:\python\PythonSpace\Data\slim\labels.txt"

# 获取所有图片及对应的分类
def get_image_and_labels(dataset_dir):
    # 图片目录
    directories = []
    labels = []
    for dir_name in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, dir_name)
        # 判断是否是目录
        if os.path.isdir(dir_path):
            directories.append(dir_path)
            labels.append(dir_name)
    # 图片路径
    image_paths = []
    for dir in directories:
        for image_name in os.listdir(dir):
            image_path = os.path.join(dir, image_name)
            image_paths.append(image_path)

    return image_path, labels
