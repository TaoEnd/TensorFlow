# coding:utf-8

import numpy as np
from tensorflow.contrib import learn

x = ["nice to meet you", "how old are you", "she is a girl"]
processor = learn.preprocessing.VocabularyProcessor(5)
x = np.array(list(processor.fit_transform(x)))
print(x)
print(len(processor.vocabulary_))
print(processor.vocabulary_._mapping)