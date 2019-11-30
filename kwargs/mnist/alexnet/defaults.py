import functools

import torch
import torch.nn as nn

from models.mnist import *

kwargs = {
    'model': AlexNet,
    'criterion': nn.CrossEntropyLoss,
    
    'epochs': 100,
    'batch': 64,
    'optimizer': functools.partial(torch.optim.SGD, lr=0.01, weight_decay=5e-4, momentum=0.9),
    'scheduler': functools.partial(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.7)
}
