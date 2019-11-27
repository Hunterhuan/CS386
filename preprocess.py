##!/usr/bin/python
# -*- coding: utf-8 -*-
# @File    : preprocess.py
# @Author  : Hunterhuan
# @Time    : 2019/11/28

import os
import pickle
import sys
import random
from collections import defaultdict
import json


random.seed(1024)

# def test():
# 	with open("val.json", "r")as f:
# 		val = json.load(f)
# 	print(val)

def main():
	dir_list = os.listdir("dataset/")
	print(dir_list)
	train_set = defaultdict(list)
	val_set = defaultdict(list)
	for dir_name in dir_list:
		pic_list = os.listdir(os.path.join("dataset",dir_name))
		random.shuffle(pic_list)
		for i in range(len(pic_list)):
			if i%5==0:
				val_set[dir_name].append(pic_list[i])
			else:
				train_set[dir_name].append(pic_list[i])
	# print(val_set.keys())
	# print(train_set.keys())
	with open("train.json","w")as f:
		json.dump(train_set, f,indent=4)
	with open("val.json","w")as f:
		json.dump(val_set, f, indent=4)
main()
# test()

# import os
# import json
# with open("train.json","r")as f:
# 	train_dataset = json.load(f)
# for dir_name, filename_list in train_dataset.items():
# 	for filename in filename_list:
# 		file_dir = os.path.join("dataset", dir_name, filename)
# 		print(file_dir)