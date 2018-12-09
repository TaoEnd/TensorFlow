python3 E:/python/PythonSpace/Data/tensorflow/models-master/research/slim/train_image_classifier.py ^
--train_dir=E:/python/PythonSpace/Data/retrain/model ^
--dataset_name=myimages ^
--dataset_split_name=train ^
--dataset_dir=E:\python\PythonSpace\Data\slim\tfrecord ^
--batch_size=10 ^
--max_number_of_steps=1000 ^
--model_name=inception_v3 ^
--clone_on_cpu=true ^
pause