# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
path = r"E:\python\PythonSpace\Data\mnist\data"
mnist = input_data.read_data_sets(path, one_hot=True)


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值，被记录在tensorboard中
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


# 命名空间
with tf.name_scope('input'):
    # 这里的none表示第一个维度可以是任意的长度
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 正确的标签
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer'):
    # 创建一个简单神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        y_ = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 二次代价函数
    loss = tf.reduce_mean(tf.square(y - y_))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中，argmax返回一维张量中最大的值所在的位置
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # 求准确率，把correct_prediction变为float32类型
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    path = r"E:\python\PythonSpace\Git\TensorFlow\Tensorboard\logs"
    writer = tf.summary.FileWriter(path, sess.graph)
    for i in range(2001):
        # m每个批次100个样本
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 执行训练的过程中，同时记录要在tensorboard中展示的值，并将结果返回
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
        # 将返回结果在tensorbard中展示
        writer.add_summary(summary, i)
        if i % 500 == 0:
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))