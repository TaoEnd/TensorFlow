# coding:utf-8

import tensorflow as tf
import numpy as np
from DataRead import get_data
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

path = r"E:\python\PythonSpace\Data\cat_vs_dog\train"
alpha = 0.4   # 数据集划分比例
img_height = 64   # 图像长宽都是64
img_width = 64

x_train, x_test, y_train, y_test = get_data(path, img_height, img_width, alpha)

x = tf.placeholder(tf.float32, [None, 64, 64, 3])
y = tf.placeholder(tf.float32, [None, 2])


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.5))

def init_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

conv1_w = init_weight([3, 3, 3, 32])
biase1 = init_biases(32)
conv2_w = init_weight([3, 3, 32, 64])
biase2 = init_biases(64)
conv3_w = init_weight([3, 3, 64, 128])
biase3 = init_biases(128)
conv4_w = init_weight([3, 3, 128, 256])
biase4 = init_biases(256)
all_w = init_weight([256*4*4, 1024])
biase5 = init_biases(1024)
out_w = init_weight([1024, 2])
biase6 = init_biases(2)


def model(x, conv1_w, conv2_w, conv3_w, conv4_w, all_w, out_w):
    layer = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding="SAME") + biase1
    conv = tf.nn.relu(layer)
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    out = tf.nn.dropout(pool, 0.5)

    layer = tf.nn.conv2d(out, conv2_w, strides=[1, 1, 1, 1], padding="SAME") + biase2
    conv = tf.nn.relu(layer)
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    out = tf.nn.dropout(pool, 0.5)

    layer = tf.nn.conv2d(out, conv3_w, strides=[1, 1, 1, 1], padding="SAME") + biase3
    conv = tf.nn.relu(layer)
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    out = tf.nn.dropout(pool, 0.5)

    layer = tf.nn.conv2d(out, conv4_w, strides=[1, 1, 1, 1], padding="SAME") + biase4
    conv = tf.nn.relu(layer)
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    out = tf.reshape(pool, [-1, all_w.get_shape().as_list()[0]])
    out = tf.nn.dropout(out, 0.5)

    all_out = tf.nn.relu(tf.matmul(out, all_w)) + biase5
    all_out = tf.nn.dropout(all_out, 0.5)

    y_pred = tf.matmul(all_out, out_w) + biase6
    return y_pred


y_pred = model(x, conv1_w, conv2_w, conv3_w, conv4_w, all_w, out_w)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)
predict_op = tf.argmax(y_pred, 1)

with tf.Session() as sess:
    batch_size = 64
    tf.global_variables_initializer().run()
    for i in range(10):
        for start in range(0, len(x_train), batch_size):
            end = start + batch_size if start + batch_size <= len(x_train) else len(x_train)
            sess.run(train_op, feed_dict={x: x_train[start: end],
                                          y: y_train[start: end]})
        y_pred = sess.run(predict_op, feed_dict={x: x_test})
        # print(y_test)
        # print(y_pred)
        accuracy = np.mean(np.argmax(y_test) == y_pred)
        print(i+1, "%.3f" % accuracy)