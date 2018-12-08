# coding:utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

path = r"E:\python\PythonSpace\Data\mnist\data"
# 因为手写数字是0-9的，所以使用one-hot编码，便于后面的softmax回归
mnist = input_data.read_data_sets(path, one_hot=True)

# 构建回归模型
# 图片大小是28*28=784像素的
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.random_normal([784, 10], stddev=0.35), name="weight")
b = tf.Variable(tf.zeros([10]), name="biases")
y_pred = tf.matmul(x, w, name="mul") + b

# 使用学习率为0.5的梯度下降算法最小化交叉熵
# 取预测值和真实值的差值的平均值

# labels表示样本的标签，它是一个向量[0.2, 0.3, 0.25, ...]，
# 每一个值表示样本属于某个类别的概率

# logits表示网络输出值，它的目的是将预测向量的概率归一化，
# 比如预测向量为[2, 3, 5]，归一化后的结果为[0.2, 0.3, 0.5]

# softmax_cross_entropy_with_logits要求labels是一个向量
# softmax_cross_entropy_with_logits_v2要求labels是一个数字，表示预测类别对应的索引
# sparse_softmax_cross_entropy_with_logits在反向传播过程中，除了对logits进行
# 反向传播外，还要对labels反向传播，但前两者只对logits进行反向传播
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
																	   labels=y))
# 使用SGD作为优化器
train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

init = tf.global_variables_initializer()
# 训练模型
with tf.Session() as sess:
	sess.run(init)
	# 训练1000次，每次循环中随机使用100个数据点替换之前的占位符
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
		if i % 100 == 0:
			# 评估模型
			# argmax：使用值最大的那个类别作为预测类别
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
			# 将布尔类型转化成浮点数，并取平均值，得到准确率
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			print("准确率：%.3f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

	# 保存文件，在tensorboard上进行可视化
	path = r"E:\python\PythonSpace\Git\TensorFlow\MNIST\logdirs"
	writer = tf.summary.FileWriter(path, sess.graph)
	writer.flush()