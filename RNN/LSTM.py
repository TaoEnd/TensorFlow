# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

path = r"E:\python\PythonSpace\Data\mnist\data"
mnist = input_data.read_data_sets(path, one_hot=True)

n_inputs = 28   # 每行有28个数据，相当于输入层中有28个神经元
n_steps = 28   # 一共28行
lstm_size = 100  # 隐层中一共有100个神经元
n_classes = 10   # 10个分类
batch_size = 50

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 定义权值
weights = {
    # 输入层与隐层之间的权值
    "in": tf.Variable(tf.random_normal([n_inputs, lstm_size])),
    # 隐层与输出层之间的权值
    "out": tf.Variable(tf.random_normal([lstm_size, n_classes]))
}

# 定义偏置
biases = {
    "in": tf.Variable(tf.constant(0.1, shape=[lstm_size, ])),
    "out": tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

# 定义LSTM网络，x表示输入
def lstm(x, weights, biases):
    # x是一个batch_size的数据，先把这些数据全部转化成28*28的数据
    x = tf.reshape(x, [-1, n_inputs])
    # 隐层计算
    cell_in = tf.matmul(x, weights["in"]) + biases["in"]
    # 将隐层计算结果转化成3维数据（batch_size, n_steps, lstm_size）
    cell_in = tf.reshape(cell_in, [-1, n_steps, lstm_size])

    # 定义LSTM的基本cell
    # forget_bias表示遗忘的比例，1.0表示全部都不遗忘，
    # state_is_tuple表示隐层中每个神经元的输出以元组形式输入下个神经元中
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    # 初始化状态，batch_size多大，就初始多少个状态
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # 计算结果
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, cell_in, initial_state=init_state)
    results = tf.matmul(final_state[1], weights["out"]) + biases["out"]

    return results


# 预测值
y_pred = lstm(x, weights, biases)
# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
# 优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
# 准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, 5001):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        if i % 1000 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))