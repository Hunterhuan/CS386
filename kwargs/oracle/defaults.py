import functools

from datasets import *
from meters import *

kwargs = {
    'dataset': functools.partial(Oracle, root='data/CIFAR'),

    'meters': {
        'acc/instance_{}': functools.partial(Meter, reduction='instance'),
    },
    'metrics': 'acc/instance_test'
}

