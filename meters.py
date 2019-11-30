# From Zhijian Liu

import numpy as np

__all__ = ['Meter']


class Meter:
    def __init__(self, reduction='instance', num_classes=None, topk=1, **kwargs):
        super(Meter, self).__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.topk = topk

        if self.reduction == 'class':
            self.count = np.ndarray((self.num_classes, self.num_classes), dtype=np.float32)
        self.reset()

    def reset(self):
        if self.reduction == 'instance':
            self.size = 0
            self.sum = 0

        if self.reduction == 'class':
            self.count.fill(0)

    def add(self, outputs, targets):
        if self.reduction == 'instance':
            _, indices = outputs.topk(self.topk, 1, True, True)

            indices = indices.transpose(0, 1)
            masks = indices.eq(targets.view(1, -1).expand_as(indices))

            self.size += targets.size(0)
            self.sum += masks[:self.topk].view(-1).float().sum(0)

        if self.reduction == 'class':
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()

            outputs = np.argmax(outputs, 1)

            x = outputs + self.num_classes * targets
            bincount = np.bincount(x.astype(np.int32), minlength=self.num_classes ** 2)

            self.count += bincount.reshape((self.num_classes, self.num_classes))

    def value(self):
        if self.reduction == 'instance':
            return self.sum / max(self.size, 1) * 100.

        if self.reduction == 'class':
            count = self.count / self.count.sum(axis=1).clip(min=1e-12)[:, None]
            return np.mean(count.diagonal()) * 100.

