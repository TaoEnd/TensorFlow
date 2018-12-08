# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
path = r"E:\python\PythonSpace\Data\mnist\data"
mnist = input_data.read_data_sets(path, one_hot=True)

# 命名空间
with tf.name_scope('input'):
    # 这里的none表示第一个维度可以是任意的长度
    x = tf.placeholder(tf.float32,[None,784], name='x-input')
    # 正确的标签
    y = tf.placeholder(tf.float32,[None,10], name='y-input')

with tf.name_scope('layer'):
    # 创建一个简单神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784,10]), name='W')
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope('softmax'):
        y_ = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 二次代价函数
    loss = tf.reduce_mean(tf.square(y - y_))
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中，argmax返回一维张量中最大的值所在的位置
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        # 求准确率，把correct_prediction变为float32类型
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    # 输出网络结构到本地
    path = r"E:\python\PythonSpace\Git\TensorFlow\Tensorboard\logs"
    writer = tf.summary.FileWriter(path, sess.graph)
    for i in range(101):
        # 每个批次100个样本
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        if i % 100 == 0:
            print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                y: mnist.test.labels}))