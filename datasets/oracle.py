import os
import pickle
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import json
import torch
import sys


class Oracle(dict):
    def __init__(self, root, **kwargs):
        super(Oracle, self).__init__({
            'train': OracleInternal(root, split='train'),
            'test': OracleInternal(root, split='test')
        })


class OracleInternal:
    def __init__(self, root, split='train', data_augmentation_type = 1):
        self.root = root
        self.split = split
        if split=="train":
            with open(os.path.join(root, "train.json"),"r")as f:
                file_json = json.load(f)
        else:
            with open(os.path.join(root, "val.json"),"r")as f:
                file_json = json.load(f)

        all_data = []
        all_labels = []
        
        label2id = {}
        label_list = list(file_json.keys())
        label_list.sort()

        for i in range(len(label_list)):
            label2id[label_list[i]] = i

        for dir_name, filename_list in file_json.items():
            # label is dir_name
            for filename in filename_list:
                file_dir = os.path.join(root, "dataset", dir_name, filename)
                # print(file_dir)
                all_data.append(np.asarray(Image.open(file_dir).resize((64,64),Image.ANTIALIAS)))
                all_labels.append(label2id[dir_name])
    
        all_data = np.vstack(all_data).reshape(-1, 1, 64, 64).transpose(0, 2, 3, 1).astype(np.uint8)
        all_labels = np.array(all_labels).astype(np.int64)
        self.all_data = all_data
        self.all_labels = all_labels
        transform_list = []
        # if data_augmentation_type==1:
        #     transform_list.append(transforms.RandomCrop(45))
        #     transform_list.append(transforms.Resize((64,64)))
        # if data_augmentation_type==2:
        #     tmp_tranforms = transforms.RandomChoice
        #     transforms_list.append(transforms.RandomResizedCrop(
        #         (64,64),
        #         scale = ()))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.1307,],
                                     std=[0.3081,]))
        if self.split == 'train':
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307,],
                                     std=[0.3081,])
            ])
    
    def __getitem__(self, index):
        '''shape: (32, 32, 3)'''
        res = np.reshape(self.all_data[index], (64,64))
        # print(res)
        # print(type(res))
        # print(res.shape)
        res = Image.fromarray(np.uint8(res))
        res = self.transform(res)
        res = torch.reshape(res, (1,64,64))
        return res, self.all_labels[index]

    def __len__(self):
        return len(self.all_data)


if __name__ == '__main__':
    dataset = Oracle('../')
    train_dataset = dataset['train']
    img = train_dataset.__getitem__(3)[0]
    img = img.numpy()
    img = (img-img.min()) / (img.max()-img.min()) * 255.0
    img = np.reshape(img, (64,64))
    img = Image.fromarray(np.uint8(img))
    img.save('test.png')