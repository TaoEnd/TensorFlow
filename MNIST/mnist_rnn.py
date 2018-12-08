# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

path = r"E:\python\PythonSpace\Data\mnist\data"
mnist = input_data.read_data_sets(path, one_hot=True)

# 设置学习率、训练次数、每轮训练的数据集大小
learning_rate = 0.01
training_times = 400
batch_size = 128

# 设置神经网络的参数
# 使用RNN分类图片时，把每张图片的行看成一个像素序列，mnist图片的大小是28*28的，
# 因此共有(28个元素的序列)*(28行)，每一步输入的序列长度为28，输入的步数是28
n_inputs = 28  # 输入层的元素个数
n_steps = 28  # 序列长度
n_hidden_units = 128  # 隐层神经元个数
n_classes = 10  # 输出层元素个数（label个数）

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
weights = {"in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
		   "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}
biases = {"in": tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
		  "out": tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}

# 定义RNN
def RNN(x, weights, biases):

	# 原本的x是三维的，维度为（batch_size, n_steps, n_inputs）
	# 首先把它转化成（batch_size * n_steps, n_inputs）的
	x = tf.reshape(x, [-1, n_inputs])

	# weights的维度是（28, 128）的，因此经过乘法操作后，
	# x_in的维度变成（128 * 28， 128）的了
	x_in = tf.matmul(x, weights["in"]) + biases["in"]

	# 将x_in的维度转变成（128，28，128）的
	x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])

	# 使用基本的LSTM
	# state_is_tuple=True时，state=(c,h)，是一个内容块c和隐态h组成的元组，
	# 否则state是一个有c和h拼接起来的张量
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

	# 将所有的c和h都初始化成0
	# 对于普通的RNN网络，初始状态的形状是:[batch_size, cell.state_size]
	init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

	# time_major决定了tensor的格式，为true时，向量的形式必须是[max_time, batch_size, depth]的
	# 为false时，向量的形式必须是[batch_size, max_time, depth]的
	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)
	results = tf.matmul(final_state[1], weights["out"]) + biases["out"]
	return results

y_pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# 计算准确率
correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1, training_times+1):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
		sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
		if i % 20 == 0:
			print(i, "%.3f" % sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))