import functools

import torch
import torch.nn as nn

from models.cifar import *

kwargs = {
    'model': resnet18,
    'num_classes': 10,
    'criterion': nn.CrossEntropyLoss,
    'init_weights': True,

    'epochs': 400,
    'batch': 64,
    'optimizer': functools.partial(torch.optim.SGD, lr=0.1, weight_decay=5e-4, momentum=0.9),
    'scheduler': functools.partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=240)
}


