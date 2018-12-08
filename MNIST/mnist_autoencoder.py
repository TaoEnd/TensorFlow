# coding:utf-8

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

# 自编码器是一个无监督模型，自编码网络的作用是将输入样本压缩到隐藏层，
# 然后解压，在输出端重建样本，最终输出层神经元数量等于输入层神经元数量

# 如果数据都是完全随机、相互独立同分布的，自编码网络就很难学习到一个
# 有效的压缩模型

# 稀疏性限制：压缩过程一方面要限制隐藏层神经元的数量，来学习一些有意义
# 的特征，另一方面还希望神经元大部分时间都是被抑制的，当神经元的输出接
# 近1时认为是被激活的，接近0时认为是被抑制的。

# 多个隐藏层的主要作用：如果输入的数据是图像，第一层会学习如何识别边，
# 第二层会学习如何去组合边，从而构成轮廓、角等，更高层会学习如何去组合
# 更有意义的特征，比如，如果输入的数据是人脸图像，更高层会学习如何识别
# 和组合眼睛、鼻子、嘴等人脸器官。

# 设置超参数
learning_rate = 0.01
training_epochs = 20  # 训练的轮数
batch_size = 256   # 每次训练的数据量
display_step = 1   # 每隔多少轮显示一次训练结果
examples_to_show = 10  # 从测试集中选择10张图片去验证自动编码器的结果

# 为该自编码网络设置两个隐藏层，第一层神经元个数为256个，第二层128个
n_hidden_1 = 256  # 第一个隐藏层神经元个数，也是特征值个数
n_hidden_2 = 128
n_input = 784   # 输入样本的特征值个数：28 * 28

# 由于是无监督学习，所有只需要样本，不需要Label
x = tf.placeholder(tf.float32, [None, n_input])

# 初始化权重和偏置
weights = {"encode_h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		   "encode_h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		   "decode_h1": tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
		   "decode_h2": tf.Variable(tf.random_normal([n_hidden_1, n_input]))}
biases = {"encode_b1": tf.Variable(tf.random_normal([n_hidden_1])),
		 "encode_b2": tf.Variable(tf.random_normal([n_hidden_2])),
		 "decode_b1": tf.Variable(tf.random_normal([n_hidden_1])),
		 "decode_b2": tf.Variable(tf.random_normal([n_input]))}

# 定义压缩函数
def encoder(x):
	# 使用sigmoid作为激活函数
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["encode_h1"]), biases["encode_b1"]))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights["encode_h2"]), biases["encode_b2"]))
	return layer_2

# 定义解压函数
def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["decode_h1"]), biases["decode_b1"]))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights["decode_h2"]), biases["decode_b2"]))
	return layer_2

# 构建模型
encoder_op = encoder(x)
x_pred = decoder(encoder_op)  # 预测值

# 定义损失函数和优化器
# 使用均方误差
cost = tf.reduce_mean(tf.pow(x_pred - x, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# 训练和评估模型
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	path = r"E:\python\PythonSpace\Data\mnist\data"
	mnist = input_data.read_data_sets(path, one_hot=True)
	total_batch = int(mnist.train.num_examples/batch_size)
	for epoch in range(training_epochs):
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={x : batch_xs})
		if epoch % display_step == 0:
			print("Epoch: %02d  cost = %.3f" % (epoch+1, c))
	print("Training Finished!")

	# 对测试集应用训练好的自编码网络
	encode_decode = sess.run(x_pred, feed_dict={x: mnist.test.images[:examples_to_show]})

	# 比较测试集原始图片和自编码网络的重建结果
	f, a = plt.subplots(2, 10, figsize=(10, 2))
	for i in range(examples_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  # 测试集
		a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))  # 重建结果
	f.show()
	plt.draw()
	plt.waitforbuttonpress()

