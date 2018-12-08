# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
path = r"E:\python\PythonSpace\Git\TensorFlow\MNIST\data"
mnist = input_data.read_data_sets(path, one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.random_normal([10]))
y_pred = tf.nn.softmax(tf.matmul(x, w) + b)

# 结果存放在一个布尔型列表中，argmax可以返回y中标签为1类别序号
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 存储模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 模型未经过训练时的准确率
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    # 载入已训练好的模型，预测图片准确率
    path = r"E:\python\PythonSpace\Git\TensorFlow\模型保存与载入\model\my_net.ckpt"
    saver.restore(sess, path)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))