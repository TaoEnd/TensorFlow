# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

path = r"D:\python\PythonSpace\Data\mnist\data"
mnist = input_data.read_data_sets(path, one_hot=True)

# 定义初始化权重函数
def init_weights(shape):
    # 初始化带有截断的权重
    weight = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(weight)

# 定义初始化偏置
def init_biases(shape):
    biase = tf.constant(0.1, shape=shape)
    return tf.Variable(biase)

# 定义卷积层
def init_conv(x, w):
    # x是一个四维向量，[batch, height, width, channels]，
    # 由批次大小，图片宽、高，以及通道数组成
    # w也是一个四维向量，[height, width, in_channels，out_channels]，
    # in_channels、out_channels分别表示输入、输出的通道数
    # strides表示步长，strides[1]、strides[2]分别表示x,y方向的移动步长
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

# 定义池化层
def init_pool(x):
    # ksize表示池化层大小
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 改变x的格式：[batch, height, width, channels]，由于不知道batch的大小，
# 所以使用-1表示
x = tf.reshape(x, [-1, 28, 28, 1])

# 设置第一层卷积
w_conv1 = init_weights([5, 5, 1, 32])
b_conv1 = init_biases([32])
# 卷积操作
h_conv1 = tf.nn.relu(init_conv(x, w_conv1) + b_conv1)
h_pool1 = init_pool(h_conv1)

# 设置第二层卷积
w_conv2 = init_weights([5, 5, 32, 64])
b_conv2 = init_biases([64])
h_conv2 = tf.nn.relu(init_conv(h_pool1, w_conv2) + b_conv2)
h_pool2 = init_pool(h_conv2)

# 经过第一二层卷积之后，图片变成了64张7*7大小的平面

# 设置第一个全连接层
w_fc1 = init_weights([7*7*64, 1024])
b_fc1 = init_biases([1024])
# 将池化层输出的结果扁平化成1维的，因为它需要与全连接层连接
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# 设置dropout时保留的比例
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 设置第二个全连接层
w_fc2 = init_weights([1024, 10])
b_fc2 = init_biases([10])

# 计算输出
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
# 优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
# 准确率
prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 64
    for i in range(1, 2001):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        if i % 500 == 0:
            accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            print("step %s, %.3f" % (i, accuracy))
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,
                                                  keep_prob: 0.5})
    print("test accuracy：", test_accuracy)