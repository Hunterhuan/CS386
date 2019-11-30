import functools

import torch
import torch.nn as nn

from models.cifar import *

kwargs = {
    'model': ResNet,
    'criterion': nn.CrossEntropyLoss,
    'depth': 56,
    'init_weights': True,
    
    
    'epochs': 300,
    'batch': 64,
    'optimizer': functools.partial(torch.optim.SGD, lr=0.1, weight_decay=5e-4, momentum=0.9),
    'scheduler': functools.partial(torch.optim.lr_scheduler.StepLR, step_size=100, gamma=0.1)
}


