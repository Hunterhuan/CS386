import torch.nn as nn
import torch
import math
from collections import OrderedDict
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torch.nn import init

__all__ = ['resnet18']

class resnet18(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained = False)
        self.model.conv1 = nn.Conv2d(1,64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
        self.model.fc = nn.Linear(512,num_classes)

    def forward(self, x):
    	return self.model(x)
