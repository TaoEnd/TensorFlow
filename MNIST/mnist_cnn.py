# coding:utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

path = r"E:\python\PythonSpace\Data\mnist\data"
mnist = input_data.read_data_sets(path, one_hot=True)
x_train, x_test, y_train, y_test = mnist.train.images, mnist.test.images, \
								   mnist.train.labels, mnist.test.labels

# set_printoptions设置打印宽度
# np.set_printoptions(linewidth=500, suppress=True)

# 将训练数据和测试数据转化成28*28的，原本是1*784的
# -1表示不考虑输入图片的数量，1表示图像是1通道的（mnist数据集是黑白的，
# 所以它是1通道的）
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# 构建一个包含3个卷积层和3个池化层，1个全连接层和1个输出层的卷积神经网络

# 首先定义初始化权重的函数
def init_weight(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.03))

# 初始化权重
w1 = init_weight([3, 3, 1, 32])  # 卷积大小是3*3的，输入维度是1， 输出维度是32
w2 = init_weight([3, 3, 32, 64])
w3 = init_weight([3, 3, 64, 128])
w4 = init_weight([128*4*4, 625]) # 全连接层，输入维度是128*4*4的，输出维度是625的
w5 = init_weight([625, 10])  # 输出层，输入维度是625的，输出维度是10的

# 构建模型
# x表示输入的数据，p_keep_conv、p_keep_hidden表示dropout时保留的比例
def model(x, w1, w2, w3, w4, w5, p_keep_conv, p_keep_hidden):
	# 第一组卷积层、池化层和dropout
	# [1, 1, 1, 1]：第一个1表示在batchsize上的滑动距离，最后一个1表示在
	# 通道上的滑动距离，中间两个1分别表示在图像长、宽上的滑动距离
	# padding表示是否对图像进行填充
	conv1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding="SAME"))
	# ksize表示池化窗口大小，第一个1和最后一个1与上面相同，中间两个2表示
	# 池化窗口大小为2*2的
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
	out1 = tf.nn.dropout(pool1, p_keep_conv)

	# 第二组
	conv2 = tf.nn.relu(tf.nn.conv2d(out1, w2, strides=[1, 1, 1, 1], padding="SAME"))
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
	out2 = tf.nn.dropout(pool2, p_keep_conv)

	# 第三组
	conv3 = tf.nn.relu(tf.nn.conv2d(out2, w3, strides=[1, 1, 1, 1], padding="SAME"))
	pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
	# 将128*4*4转化成1*2048的
	out3 = tf.reshape(pool3, [-1, w4.get_shape().as_list()[0]])
	out3 = tf.nn.dropout(out3, p_keep_conv)
	# 第三组结束后图像就变成128*4*4的了

	#全连接层
	out4 = tf.nn.relu(tf.matmul(out3, w4))
	out4 = tf.nn.dropout(out4, p_keep_hidden)

	# 输出层
	y_pred = tf.matmul(out4, w5)
	return y_pred

p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)
y_pred = model(x, w1, w2, w3, w4, w5, p_keep_conv, p_keep_hidden)

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
# RMSPropOptimizer是一种自适应的参数更新方法，根据参数的重要性对不同参数进行成
# 不同程度的更新，它是Adagrad的改进版，后者可以对低频的参数做较大的更新，
# 对高频的参数做较小的更新，因此这种方法对稀疏数据有较好的表现，提供了SGD的鲁棒性.
# RMSPropOptimizer解决了Adagrad学习率急剧下降的问题，通过对不同维度进行不同程度的
# 加权，如果某一维度的导数比较大，则平均指数加权就大，如果某一维度的导数比较小，则
# 平均指数加权就小，这样保证了每个维度的导数都在同一个量级，进而减少了摆动
# 0.001表示学习率，0.9表示衰减值
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(y_pred, 1)

# 训练模型和评估模型
with tf.Session() as sess:
	batch_size = 128
	test_size = 256
	tf.global_variables_initializer().run()
	for i in range(10):
		training_batch = zip(range(0, len(x_train), batch_size),
							 range(batch_size, len(x_train)+1, batch_size))
		for start, end in training_batch:
			sess.run(train_op, feed_dict={x: x_train[start: end],
										  y: y_train[start: end],
										  p_keep_conv: 0.8,
										  p_keep_hidden: 0.5})
		# 每次随机选择256个样本进行测试
		test_indexs = np.arange(len(x_test))
		np.random.shuffle(test_indexs)
		test_indexs = test_indexs[0: test_size]
		y_pred = sess.run(predict_op, feed_dict={x: x_test[test_indexs],
												 p_keep_conv: 1.0,
												 p_keep_hidden: 1.0})
		print(i+1, "%.3f" % np.mean(np.argmax(y_test[test_indexs], axis=1)==y_pred))
