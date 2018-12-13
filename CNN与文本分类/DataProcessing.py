# coding:utf-8

import numpy as np
import re

# 数据清洗
def clean_data(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# 加载数据
def load_data_and_labels(pos_data_path, neg_data_path):
    pos_data = open(pos_data_path, "rb").read().decode("utf-8")
    neg_data = open(neg_data_path, "rb").read().decode("utf-8")
    # 文件中的每一行代表一封邮件，去除最后一个空行
    pos_lines = pos_data.split("\n")[:-1]
    neg_lines = neg_data.split("\n")[:-1]
    pos_lines = [pos.strip() for pos in pos_lines]
    neg_lines = [neg.strip() for neg in neg_lines]
    # 样本
    x = pos_lines + neg_lines
    x = [clean_data(line) for line in x]
    pos_labels = [[0, 1] for _ in pos_lines]
    neg_labels = [[1, 0] for _ in neg_lines]
    y = np.concatenate([pos_labels, neg_labels], axis=0)
    return [x, y]

# 获得每一个batch的数据
def get_batch_data(x_train, y_train, batch_size, shuffle=True):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    data_size = len(x_train)
    batch_nums = int((data_size-1)/batch_size)+1
    if shuffle:
        shuffle_index = np.random.permutation(np.arange(data_size))
        shuffled_x = x_train[shuffle_index]
        shuffled_y = y_train[shuffle_index]
    else:
        shuffled_x = x_train
        shuffled_y = y_train
    for batch_num in range(batch_nums):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, batch_size)
        # 得到一个关于每个batch的生成器，并返回该生成器
        yield shuffled_x[start_index: end_index], shuffled_y[start_index: end_index]