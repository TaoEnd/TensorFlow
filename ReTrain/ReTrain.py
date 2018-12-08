# coding:utf-8

# 使用重新训练的inception_v3模型来分类

import tensorflow as tf
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# 读自定义的图片类别
label_path = r"E:\python\PythonSpace\Data\retrain\output_labels.txt"
with open(label_path, "r") as fr:
    lines = fr.readlines()
    uid_to_label = {}
    for uid, label in enumerate(lines):
        uid_to_label[uid] = label.strip("\n")

def uid_to_label(uid):
    if uid not in uid_to_label:
        return ""
    return uid_to_label[uid]

# 创建一个图来存放训练好的模型
model_path = r"E:\python\PythonSpace\Data\retrain\output_graph.pb"
with tf.gfile.FastGFile(model_path, "rb") as fr:
    graph = tf.GraphDef()
    graph.ParseFromString(fr.read())
    tf.import_graph_def(graph, name="")

with tf.Session() as sess:
    # 根据名字返回tensor数据
    # <name>:0，形如“conv1”是节点名称，而“conv1:0”是张量名称，
    # 表示节点的第一个输出张量
    tensor = sess.graph.get_tensor_by_name("final_result:0")
    test_image_path = r"E:\python\PythonSpace\Git\TensorFlow\ReTrain\test_data"
    for root, dirs, files in os.walk(test_image_path):
        for file in files:
            image_path = os.path.join(root, file)
            image_data = tf.gfile.FastGFile(image_path, "rb").read()
            predictions = sess.run(tensor, {"DecodeJpeg/contents:0": image_data})
            predictions = np.squeeze(predictions)

            # 显示图片
            image = Image.open(image_path)
            plt.show(image)
            plt.axis("off")
            plt.show()

            # 排序
            top_k = predictions.argsort()[::-1]
            print(top_k)
            for uid in top_k:
                label = uid_to_label[uid]
                score = predictions[uid]
                print("%s（score = %.3f）" % (label, score))
            print("--------------------")
