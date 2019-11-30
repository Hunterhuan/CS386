import functools

import torch
import torch.nn as nn

from models.cifar import *

kwargs = {
    'model': alexnet,
    'num_classes': 40,
    'criterion': nn.CrossEntropyLoss,
    'depth': 19,
    'init_weights': True,

    'epochs': 240,
    'batch': 64,
    'optimizer': functools.partial(torch.optim.SGD, lr=0.1, weight_decay=5e-4, momentum=0.9),
    'scheduler': functools.partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=240)
}


