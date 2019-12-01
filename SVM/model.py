import os
import json
import numpy as np
from sklearn import svm


def load_data():
    train_data = np.loadtxt("traindataset/traindata.txt", dtype=int)
    train_x = train_data[:, :-1]
    train_x = np.transpose(train_x)
    train_x = np.transpose(train_x)
    train_y = train_data[:, -1]
    val_data = np.loadtxt("valdataset/valdata.txt", dtype=int)
    SVM = svm.SVC(gamma="scale")
    SVM.fit(train_x, train_y)
    val_x = val_data[:, :-1]
    val_x = np.transpose(val_x)
    val_x = np.transpose(val_x)
    val_y = val_data[:, -1]
    result = SVM.predict(val_x)
    x = result - val_y
    print(len(np.argwhere(result == val_y)) / len(result))



load_data()