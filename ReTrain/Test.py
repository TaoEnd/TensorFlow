import os

string = "%s_%02d.tfrecord"
result = string % ("a", 3)
print(result)

LABELS_PATH = r"E:\python\PythonSpace\Data\slim\labels.txt"
if not os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "w") as fw:
        fw.write("a\n")