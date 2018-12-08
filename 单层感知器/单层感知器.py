# coding:utf-8

import numpy as np
from matplotlib import pylab as plt

# 假设平面上有三个点，（3,3）、（4,3）的标签是1，（1,1）的标签是-1，
# 构建神经网络来分类
# 将偏置放到系数中，作为w0，因此可以将输入看成（1,3,3）、（1,4,3）、（1,1,1），
# 它们对应的label分别为（1,1,-1）

x = np.array([[1, 3, 3],
              [1, 4, 3],
              [1, 1, 1]])
y = np.array([1, 1, -1])

w = (np.random.random(3)-0.5)*2  # 初始权重，范围在-1到1之间
alpha = 0.2   # 学习率
y_pred = 0


def update():
    global x, y, w, alpha, y_pred
    y_pred = np.sign(np.dot(x, w.T))
    delta_w = alpha*np.dot(y-y_pred, x)/x.shape[0]
    w = w + delta_w


for i in range(100):
    update()
    print(i, w)
    if (y_pred == y).all():
        print("Finished")
        break

k = -w[1]/w[2]
b = -w[0]/w[2]
print(k, b)
x = np.linspace(0, 5, 50)
y = k*x + b
plt.plot(x, y, "r")
plt.plot([3, 4], [3, 3], "bo")
plt.plot([1], [1], "go")
plt.show()