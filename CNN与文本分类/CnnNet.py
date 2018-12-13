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
        self.l2_loss = tf.constant(0.0)

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

        # 定义卷积层和池化层
        # 邮件在处理成一个矩阵时，每一个邮件对应一个矩阵，矩阵的宽是邮件中每个词的
        # 词向量长度，矩阵的长是邮件中词的个数，为了使得所有邮件矩阵的大小相同，
        # 因此规定所有矩阵的长为单词数量最多的邮件对应的单词数量
        pooled_outputs = []
        for i, conv_size in enumerate(conv_sizes):
            with tf.name_scope("conv-maxpool-%d" % conv_size):
                conv_shape = [conv_size, embedding_size, 1, num_convs]
                # truncated_normal()：利用3sigma原则，从截断的正态分布中输出随机值，
                # 当随机值大于或者小于2倍偏差值时，就丢弃该随机值，重新选择一个
                # stddev代表标准差
                w = tf.Variable(tf.truncated_normal(conv_shape, stddev=0.3), name="w")
                b = tf.Variable(tf.random_normal(shape=[num_convs], stddev=0.1), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, w, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")
                # 将向量b加到矩阵conv的每一行，b与conv中每一行的长度相等
                relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # ksize中第二个参数表示height，第三个参数表示width
                pooled = tf.nn.max_pool(relu, ksize=[1, sequence_length-conv_size+1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="pool")
                pooled_outputs.append(pooled)

        num_convs_total = num_convs * len(conv_sizes)
        # 在pooled_outputs的第四维上进行连接
        self.pool = tf.concat(pooled_outputs, 3)
        # 把原本重叠在一起的卷积层全部竖起来放，并且连接在一起，以便后面进行全连接操作
        self.pool_flat = tf.reshape(self.pool, [-1, num_convs_total])

        # dropout
        with tf.name_scope("dropout"):
            self.dropout = tf.nn.dropout(self.pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            w = tf.Variable(tf.truncated_normal(shape=[num_convs_total, num_classes],
                                                stddev=0.3), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.y_pred = tf.nn.xw_plus_b(self.dropout, w, b, name="scores")
            self.l2_loss += tf.nn.l2_loss(w)
            self.l2_loss += tf.nn.l2_loss(b)

        with tf.name_scope("cost"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred,
                                                                          labels=self.y))
            self.cost = cost + l2_lamda * self.l2_loss

        with tf.name_scope("accuracy"):
            self.predictions = tf.equal(tf.argmax(self.y_pred, 1),tf.argmax(self.y, 1),
                                        name="predicitons")
            self.accuracy = tf.reduce_mean(tf.cast(self.predictions, tf.float32))
