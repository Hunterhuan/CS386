import functools

import torch
import torch.nn as nn

from models.mnist import *

kwargs = {
    'model': VGG,
    'criterion': nn.CrossEntropyLoss,
    'depth': 16,
    'init_weights': True,

    'epochs': 24,
    'batch': 16,
    'optimizer': functools.partial(torch.optim.SGD, lr=0.001, weight_decay=5e-4, momentum=0.9),
    'scheduler': functools.partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=240)
}


