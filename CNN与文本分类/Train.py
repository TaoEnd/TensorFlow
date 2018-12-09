# coding:utf-8

import tensorflow as tf
from CNN与文本分类 import DataProcessing as dataprocessing
from tensorflow.contrib import learn
import numpy as np
from CNN与文本分类.CnnNet import CnnNet

# 定义网络中需要使用的参数
# 第一个参数是参数的名字，第二个参数是参数的默认值，第三个是参数的描述
tf.flags.DEFINE_float("sample_ratio", 0.1, "测试样本所占的比例")
pos_path = r"E:\python\PythonSpace\Data\cnn_vs_text\data\rt-polarity.pos"
neg_path = r"E:\python\PythonSpace\Data\cnn_vs_text\data\rt-polarity.neg"
tf.flags.DEFINE_string("pos_data_path", pos_path, "正样本地址")
tf.flags.DEFINE_string("neg_data_path", neg_path, "负样本（垃圾邮件）地址")

# 模型超参数
tf.flags.DEFINE_integer("word_vec_dim", 128, "词向量长度")
tf.flags.DEFINE_string("conv_sizes", "3,4,5", "卷积核宽度")
tf.flags.DEFINE_integer("num_convs", 128, "每类卷积核的个数")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout保留的比例")
tf.flags.DEFINE_float("l2_lamda", 0.0, "L2正则化项系数")

# 训练模型使用的参数
tf.flags.DEFINE_integer("batch_size", 64, "batch_size")
tf.flags.DEFINE_integer("num_epoches", 200, "训练轮数")
tf.flags.DEFINE_integer("evaluate_every", 50, "每隔多少轮输出一次模型预测值")
tf.flags.DEFINE_integer("checkpoint_every", 100, "每隔多少轮保存一次模型")
tf.flags.DEFINE_integer("num_checkpoints", 5, "最多保存的模型个数，多余的模型会覆盖前面的模型")

# session的配置参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "当指定设备不存在时，是否自动分配其它设备")
tf.flags.DEFINE_boolean("log_device_placement", False, "是否打印分配日志")

# 打印参数
flags = tf.flags.FLAGS
params = flags.flag_values_dict()
for param in sorted(params):
    value = params[param]
    print(param, value)

# 加载数据
print("-------Loading data...-------")
x, y = dataprocessing.load_data_and_labels(params["pos_data_path"], params["neg_data_path"])
# 为了保证每个邮件的单词个数是一样的，需要找到这些邮件中单词个数的最大值，
# 对于那些单词个数小于最大值的邮件，使用0进行填充，类似于padding操作

# VocabularyProcessor()：根据所有已分词号的文本建立一个词典，
# 然后找出一句话中每个词在词典中对应的索引，当这句话的长度小于max_document_length时，
# 就在后面补0

# x最终的宽度等于这些邮件中单词个数最多的那封邮件中包含的单词个数
max_document_length = max([len(s.split(" ")) for s in x])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x)))

# shuffle操作
# shuffle之后需要使用的是新的打乱的index顺序，通过np.random.permutation
# 就可以得到打乱顺序的index数组
np.random.seed(0)
shuffle_index = np.random.permutation(np.arange(len(x)))
x_shuffled = x[shuffle_index]
y_shuffled = y[shuffle_index]

# 样本切分
num_test = int(params["sample_ratio"]*len(x))
x_test, x_train = x_shuffled[: num_test], x_shuffled[num_test: ]
y_test, y_train = y_shuffled[: num_test], y_shuffled[num_test: ]
print("all data's num：", len(x))
print("test data's num：", num_test)

with tf.Graph().as_default():
    # 配置session参数
    session_conf = tf.ConfigProto(allow_soft_placement=params["allow_soft_placement"],
                                  log_device_placement=params["log_device_placement"])
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CnnNet(sequence_length=x_train.shape[1], # 所有邮件中，单词个数最多的邮件中的包含的单词个数
                     num_classes=y_train.shape[1], # 类别数量
                     # vocab_processor.vocabulary_：返回的是一个字典，这个字典中是每个单词
                     # 与序号的对应关系，字典的长度是单词数加1，因为字典中还包含一个“<UNK>”
                     vocab_size=len(vocab_processor.vocabulary_),
                     embedding_size=params["word_vec_dim"], # 词向量长度
                     conv_sizes=params["conv_sizes"], # 有多少个不同大小的卷积核
                     num_convs=params["num_convs"], # 每一类不同大小的卷积核中，包含多少个卷积核
                     l2_lamda=params["l2_lamda"])


