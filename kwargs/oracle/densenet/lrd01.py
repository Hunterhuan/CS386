import functools

import torch
import torch.nn as nn

kwargs = {

    'optimizer': functools.partial(torch.optim.SGD, lr=0.01, weight_decay=5e-4, momentum=0.9)
}

