# coding:utf-8

import tensorflow as tf

# 1、创建图和启动图
#   Tensorflow中每一个节点代表一个操作（简称op，常量也是一个op），
#   节点必须定义在图中，图必须在会话中启动，使用Session创建会话，
#   Tensorflow的Session中默认包含一个图，绝大部分程序都可以在这个
#   默认的图中实现，使用sess.run()方法执行程序

# 2、交互式使用和变量
# 2.1、交互式会话
#   不使用交互式会话时，需要使用run()方法执行会话，使用交互式会话时
#   可以不用使用run()方法，改为直接使用eval()方法

# 使用交互式会话时，不能使用 with tf.InteractiveSession() sess 方式
sess =  tf.InteractiveSession()
x = tf.Variable([2, 3])
a = tf.constant([1, 2])
# 使用变量之前需要进行初始化，常量不需要
x.initializer.run()
result = tf.subtract(x, a)
print(result.eval())

# 3、fetch和feed操作
# fetch：在一个run中执行多个操作，
#   result中的第一个元素是add的结果，mul的第二个元素是mul的结果
state1 = tf.constant(1)
state2 = tf.constant(2)
state3 = tf.constant(3)
add = tf.add(state1, state2)
mul = tf.multiply(state3, add)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run([add, mul])
    print(result)

# feed：使用feed时，必须存在占位符，placeholder
# 它就是run([...], feed_dict={...})

