# coding:utf-8

import numpy as np
import tensorflow as tf

# 拟合曲线y = x^2 - 0.5
# 生成数据
x_data = np.linspace(-1, 1, 300).reshape(300, 1)
# 生成满足均值为0，方差为0.05的正态分布的噪声数据
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 占位符
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 构建一个隐藏层和一个输出层，作为神经网络中的层，输入参数应该有4个变量，
# 即：输入数据、输入数据的维度、输出数据的维度和激活函数，每一层经过
# 向量化（y = w * x + b）处理，并且经过激活函数的非线性处理后，得到最终结果
def add_layer(inputs, in_size, out_size, activation_function=None):
	# 使用正态分布随机初始化权重矩阵
	weights = tf.Variable(tf.random_normal([in_size, out_size]))
	# 初始化偏置矩阵
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	# 矩阵相乘
	y_hat = tf.matmul(inputs, weights) + biases
	if activation_function is None:
		outputs = y_hat
	else:
		outputs = activation_function(y_hat)
	return outputs

# 构建隐藏层，假设隐藏层有20个神经元
hidden_layer = add_layer(xs, 1, 20, activation_function=None)
# 构建输出层，假设输出层和输入层一样，有1个神经元
prediction_layer = add_layer(hidden_layer, 20, 1, activation_function=None)

# 构建损失函数，使用输出层的预测值与真实值之间的MSE作为损失函数，
# reduction_indices=[1]表示将每一行向量压缩成一个数（加和），
# 最终是m*1维的矩阵，reduction_indices=[0]则表示将每一列向量
# 压缩成一个数（加和），最终是1*m维的矩阵
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction_layer),
									reduction_indices=[1]))
# 使用梯度下降法，以0.1的学习率进行学习
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 训练模型
init = tf.global_variables_initializer()  # 初始化所有变量
# 开启会话
with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):    # 训练1000次
		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
		if i % 50 == 0:  # 每隔50次就打印一次损失函数值
			print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))