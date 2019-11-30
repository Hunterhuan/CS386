import functools

from datasets import *
from meters import *

kwargs = {
    'dataset': MNIST,

    'meters': {
        'acc/instance_{}': functools.partial(Meter, reduction='instance'),
    },
    'metrics': 'acc/instance_test'
}

