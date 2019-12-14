import os
import json
from PIL import Image
import numpy as np


def load_data():
    with open("train.json", "r")as f:
        train_set = json.load(f)
    with open("val.json", "r")as f:
        val_set = json.load(f)
    if not os.path.exists("SVM/traindataset"):
        os.makedirs("SVM/traindataset")
    for key in train_set.keys():
        for pic in train_set[key]:
            picpath = "dataset_binary/{0}/{1}".format(key, pic)
            img = Image.open(picpath, "r")
            img = data_preprocess(img)
            with open("SVM/traindataset/traindata.txt", "a+") as f:
                for i in img:
                    f.write("{0}\t".format(i))
                f.write("{0}\n".format(int(key)))
    if not os.path.exists("SVM/valdataset"):
        os.makedirs("SVM/valdataset")
    for key in val_set.keys():
        for pic in val_set[key]:
            picpath = "dataset_binary/{0}/{1}".format(key, pic)
            img = Image.open(picpath, "r")
            img = data_preprocess(img)
            with open("SVM/valdataset/valdata.txt", "a+") as f:
                for i in img:
                    f.write("{0}\t".format(i))
                f.write("{0}\n".format(int(key)))


def data_preprocess(img):
    img = img.resize((32, 32))
    img = np.array(img)
    img = img.reshape(1, -1)[0]
    return img


load_data()