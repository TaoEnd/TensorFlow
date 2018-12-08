# coding:utf-8

import numpy as np
import os
from matplotlib import pylab as plt
from PIL import Image


path = r"E:\python\PythonSpace\Data\cat_vs_dog\train"
files = os.listdir(path)
path = os.path.join(path, files[0])
cat = plt.imread(path)
img = Image.open(path)
img = img.resize((128, 128))
img = np.array(img)
print(type(img))
# plt.imshow(cat)
# plt.imshow(img)
# plt.show()

print(int(5.4))

li1 = np.array([1, 2, 3, 4, 5])
li2 = np.array([1, 0, 0, 1, 0])
np.random.seed(1)
np.random.shuffle(li1)
np.random.seed(1)
np.random.shuffle(li2)
print(li1)
print(li2)

li1 = np.array([[1, 2], [3, 4]])
li2 = np.array([[1, 2], [3, 4]])
print(li1)
li3 = np.append(li1, li2, axis=0)
print(li3)

li1 = [[1, 2], [3, 4]]
li2 = [[3, 4]]
li3 = []
li3.extend(li1)
li3.extend(li2)
print(li3)
for i in range(3, 11, 3):
    print(i)


