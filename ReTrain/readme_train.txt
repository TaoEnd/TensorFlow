Slim_VS_TFRecord文件用来生成tfrecord文件和label文件
myimages.py用来读取tfrecord文件
使用slim之前，先要生成tfrecord文件，还要修改slim中的train_image_classifier.py文件
train.bat中的参数：
    train_dir：表示自己训练生成的模型存在的文件夹
    dataset_name：表示在train_image_classifier.py中使用哪个文件
    dataset_split_name：表示使用哪个类型（train、test）的数据进行训练
    dataset_dir：表示之前生成的tfrecord文件的存放地址，需要把生成的label文件也放在这个文件夹中
    batch_size：每次处理的图片大小
    max_number_of_steps：最大迭代次数
    model_name：表示使用的模型的名字（inception_v2、inception_v3等）
    clone_on_cpu=true：开启cpu选项，否则有时可能报错

参考：https://blog.csdn.net/gubenpeiyuan/article/details/80284888