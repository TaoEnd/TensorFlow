# coding:utf-8

# 使用slim来从头开始构建模型

import tensorflow as tf
import os
import random
import math
import sys

# 测试集数据
NUM_TEST = 500
# 随机种子
RANDOM_SEED = 0
# 数据块，当样本很大时，需要将其划分成多个块
NUM_SHARDS = 5
# tfrecord文件命名格式
TFRECORD_NAME = "%s_%02d.tfrecord"
# 数据集路径
DATASET_DIR = r"E:\python\PythonSpace\Data\slim\picture"
# label文件位置，该文件中是数字和label的对应关系
LABELS_PATH = r"E:\python\PythonSpace\Data\slim\labels.txt"
# tfrecord文件位置，该文件与模型训练相关
TFRECORD_DIR = r"E:\python\PythonSpace\Data\slim\tfrecord"

# 获取所有图片及对应的分类
def get_image_and_labels(dataset_dir):
    # 图片目录
    directories = []
    labels = []
    for dir_name in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, dir_name)
        # 判断是否是目录
        if os.path.isdir(dir_path):
            directories.append(dir_path)
            labels.append(dir_name)
    # 图片路径
    image_paths = []
    for dir in directories:
        for image_name in os.listdir(dir):
            image_path = os.path.join(dir, image_name)
            image_paths.append(image_path)

    return image_paths, labels

# 判断tfrecord文件是否存在，模型训练时需要使用这种格式的文件
def tfrecord_exists(tfrecord_dir):
    file_names = os.listdir(tfrecord_dir)
    for type in ["train", "test"]:
        for index in range(NUM_SHARDS):
            tfrecord_name = TFRECORD_NAME % (type, index)
            if tfrecord_name not in file_names:
                return False
    return True

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# 根据图片生成tfrecord文件
def image_to_tfrecord(image_data, image_format, label_id):
    tfrecord = tf.train.Example(features=tf.train.Features(
        feature = {
            "image/encoded": bytes_feature(image_data),
            "image/format": bytes_feature(image_format),
            "image/class/label": int64_feature(label_id)
        }
    ))
    return tfrecord

# 将数据转化成tfrecord格式
def tfrecord_convert(type, images, labels_vs_ids, tfrecord_dir):
    assert type in ["train", "test"]
    # 计算每个数据块中包含的样本数
    num_per_shared = int(len(images)/NUM_SHARDS)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(NUM_SHARDS):
                tfrecord_name = TFRECORD_NAME % (type, shard_id)
                tfrecord_path = os.path.join(tfrecord_dir, tfrecord_name)
                with tf.python_io.TFRecordWriter(tfrecord_path) as tfwriter:
                    # 每个样本开始的位置
                    start_index = shard_id * num_per_shared
                    # 每个样本结束的位置
                    end_index = min((shard_id+1)*num_per_shared, len(images))
                    for i in range(start_index, end_index):
                        # 由于读取的图片中可能存在损坏的，因此使用try...catch
                        try:
                            # sys.stdout.write("Covertin：shard %02d, image %03d" % (shard_id, i+1))
                            # sys.stdout.flush()
                            print("Coverting：shard %02d, image %04d" % (shard_id, i+1))
                            # 读取图片
                            image_data = tf.gfile.FastGFile(images[i], "rb").read()
                            # 获得图片的类别名称
                            label = os.path.basename(os.path.dirname(images[i]))
                            # 获得类别对应的id
                            id = labels_vs_ids[label]
                            # 生成tfrecord文件
                            tfrecord = image_to_tfrecord(image_data, b"jpg", id)
                            tfwriter.write(tfrecord.SerializeToString())
                        except IOError as e:
                            print("Could not read：", images[i])
                            print(e)
                            print()
    sys.stdout.write("\n")
    sys.stdout.flush()

def write_label(labels_vs_ids, file_path):
    fw = open(file_path, "w")
    for label in labels_vs_ids:
        id = labels_vs_ids[label]
        fw.write("%d:%s\n" % (id, label))


if __name__ == "__main__":
    # 判断tfrecord文件是否存在
    if tfrecord_exists(TFRECORD_DIR):
        print("tfrecord has exists")
    else:
        # 获得所有图片及分类
        image_paths, labels = get_image_and_labels(DATASET_DIR)
        # 将分类转化成字典格式，类似于：{"animal": 0, "airplane": 1,...}
        labels_vs_ids = dict(zip(labels, range(len(labels))))

        # 将数据划分成训练集和测试集
        random.seed(RANDOM_SEED)
        random.shuffle(image_paths)
        training_images = image_paths[NUM_TEST: ]
        testing_images = image_paths[: NUM_TEST]

        # 将图片转换成tfrecord文件
        tfrecord_convert("train", training_images, labels_vs_ids, TFRECORD_DIR)
        tfrecord_convert("test", testing_images, labels_vs_ids, TFRECORD_DIR)

        # 输出label文件
        print(labels_vs_ids)
        write_label(labels_vs_ids, LABELS_PATH)