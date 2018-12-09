# coding:utf-8

import tensorflow as tf
import numpy as np

class CnnNet(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, conv_sizes, num_convs, l2_lamda):
        # 当输入的是一个batch的数据时，需要使用[... , ...]，如果仅仅只是一个数，
        # 那么就不需要[... , ...]
        self.x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="labels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device("/gpu:0"), tf.name_scope("embedding"):
            # 初始化权重
            self.w = tf.Variable(tf.random_normal([vocab_size, embedding_size], stddev=0.5),
                                 name="weights")
            # w是字典库中所有单词的权重组成的矩阵，每个单词的权重是一个长度为
            # embedding_size长度的向量，x中的元素表示一个邮件中的单词构成的向量，
            # 向量中的元素是数字，表示当前这个单词在字典库中的编号，
            # embedding_lookup()的作用：比如x=[1, 3]，就表示从w中选择第二行和
            # 第四行来构成一个新的矩阵
            self.embedded_chars = tf.nn.embedding_lookup(self.w, self.x)
            # 扩展维度，-1表示在最后一维上再增加一维
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)