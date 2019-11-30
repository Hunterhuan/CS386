import functools

import torch
kwargs = {
    'scheduler': functools.partial(torch.optim.lr_scheduler.StepLR, step_size=100, gamma=0.1)
}