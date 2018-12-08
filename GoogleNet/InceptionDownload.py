# coding:utf-8

import tensorflow as tf
import os
import tarfile
import requests

# inception模型下载地址
url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
model_dir = r"E:\python\PythonSpace\Data\inception\model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 获取文件名和文件路径
filename = url.split("/")[-1]
filepath = os.path.join(model_dir, filename)

# 下载模型
if not os.path.exists(filepath):
    print("download：", filename)
    r = requests.get(url, stream=True)
    with open(filepath, "wb") as fw:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                fw.write(chunk)
    print("finished：", filename)

# 解压文件
tarfile.open(filepath, "r:gz").extractall(model_dir)

# 模型结构存放文件
log_dir = r"E:\python\PythonSpace\Data\inception\log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classify_image_graph_def.pb为google训练好的模型
graph_file = os.path.join(model_dir, "classify_image_graph_def.pb")
with tf.Session() as sess:
    # 创建一个图来存放该模型
    with tf.gfile.FastGFile(graph_file, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        tf.import_graph_def(graph_def, name="")
    # 保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()
