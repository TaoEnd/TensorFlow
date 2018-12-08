# coding:utf-8

# inception-v3是googlenet的第三个版本，
# 使用该网络识别图片

import tensorflow as tf
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class NodeLookup(object):
    def __init__(self):
        label_path = r"E:\python\PythonSpace\Data\inception\model\imagenet_2012_challenge_label_map_proto.pbtxt"
        uid_path = r"E:\python\PythonSpace\Data\inception\model\imagenet_synset_to_human_label_map.txt"
        self.node_lookup = self.load(label_path, uid_path)

    def load(self, label_path, uid_path):
        # 加载分类字符串文件（分类id与类别描述的对应字符串）
        proto_lines = tf.gfile.GFile(uid_path).readlines()
        uid_vs_describe = {}
        # 读取数据
        for line in proto_lines:
            items = line.strip("\n").split("\t")
            # 分类编号与分类描述
            uid = items[0]
            describe = items[1]
            uid_vs_describe[uid] = describe

        # 加载分类编号与数字之间对应的文件
        label_lines = tf.gfile.GFile(label_path).readlines()
        id_vs_uid = {}
        for line in label_lines:
            if line.strip().startswith("target_class:"):
                id = int(line.split(":")[1].strip())
            if line.strip().startswith("target_class_string:"):
                uid = line.split(":")[1].strip()[1:-1]
                id_vs_uid[id] = uid

        # 找到数字与图片描述之间的对应关系
        id_vs_describe = {}
        for id, uid in id_vs_uid.items():
            describe = uid_vs_describe[uid]
            id_vs_describe[id] = describe
        return id_vs_describe

    # 根据id返回图片描述，因为goolenet的预测结果是id，
    # 需要根据id来找到图片的描述信息
    def id_to_describe(self, id):
        if id not in self.node_lookup:
            return "There has not same picture!"
        return self.node_lookup[id]

# 创建一个图来存放googlenet
model_path = r"E:\python\PythonSpace\Data\inception\model\classify_image_graph_def.pb"
with tf.gfile.FastGFile(model_path, "rb") as fr:
    graph = tf.GraphDef()
    graph.ParseFromString(fr.read())
    tf.import_graph_def(graph, name="")

with tf.Session() as sess:
    tensor = sess.graph.get_tensor_by_name("softmax:0")
    test_data_path = r"E:\python\PythonSpace\Git\TensorFlow\GoogleNet\data"
    for root, dirs, files in os.walk(test_data_path):
        # 载入图片
        for file in files:
            image = tf.gfile.FastGFile(os.path.join(test_data_path, file), "rb").read()
            # 图片格式是.jpg的
            predictions = sess.run(tensor, {"DecodeJpeg/contents:0": image})
            # 把二维图片转化成一维的
            predictions = np.squeeze(predictions)

            # 打印图片路径及名称
            image_path = os.path.join(root, file)
            print(image_path)
            # 显示图片
            image = Image.open(image_path)
            plt.imshow(image)
            plt.axis("off")
            plt.show()

            # 排序
            # predictions返回的是一个1000维的向量，因为GoogleNet是对1000个
            # 类别的图片进行分类，使用的是softmax的方式，所以predictions返回
            # 当前预测的图片在每个类别上的概率
            # 取概率最大的5个类别，并从大到小进行降序排列，[-5：]表示取最大的
            # 5个类别，[::-1]表示对5个类别的概率进行降序排列
            # top_k中保存的是每个类别对应的数字序号
            top_k = predictions.argsort()[-5:][::-1]
            node_lookup = NodeLookup()
            for id in top_k:
                desribe = node_lookup.id_to_describe(id)
                # 获取每个分类的置信度
                score = predictions[id]
                print("%s (score = %.3f)" % (desribe, score))
            print("----------------")
