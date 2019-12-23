import argparse
import importlib
import os
import random

import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kwargs', default=None)
    parser.add_argument('--devices', default=None)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    args.kwargs = args.kwargs.replace('/', '.').replace('.py', '')

    # load kwargs
    kwargs = {}
    prefix = ''
    for name in args.kwargs.split('.')[:-1]:
        prefix += name + '.'
        kwargs = {**kwargs, **importlib.import_module(prefix + 'defaults').kwargs}
    kwargs = {**kwargs, **importlib.import_module(args.kwargs).kwargs}

    print('==> parsed arguments')
    for k, v in kwargs.items():
        print('[{}] = {}'.format(k, v))

    # device = 'cuda'
    # set devices
    if args.devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
        cudnn.benchmark = True
        device = 'cuda'
    else:
        device = 'cpu'

    resume = False
    if args.resume is not None:
        if args.resume == "1":
            resume = True


    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # dataset
    print('==> loading dataset "{}"'.format(kwargs['dataset']))
    dataset = kwargs['dataset'] = kwargs['dataset'](**kwargs)

    # loaders
    loaders = {}
    for split in ['train', 'test']:
        loaders[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=kwargs['batch'],
            shuffle=(split == 'train'),
            num_workers=kwargs['workers'],
            pin_memory=True
        )

    # create model
    print('==> creating model "{}"'.format(kwargs['model']))
    model = kwargs['model'] = kwargs['model'](**kwargs).to(device)

    # define loss function, optimizer and scheduler
    criterion = kwargs['criterion'] = kwargs['criterion']().to(device)
    optimizer = kwargs['optimizer'] = kwargs['optimizer'](model.parameters())

    # can choose not to use scheduler
    if kwargs['scheduler'] is not None:
        scheduler = kwargs['scheduler'] = kwargs['scheduler'](optimizer)


    # define save path and summary writer
    save_path = os.path.join('exp', args.kwargs[7:])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    elif os.path.exists(os.path.join(save_path, 'latest.pth')) and resume:
        kwargs['resume'] = os.path.join(save_path, 'latest.pth')


    epoch, best = 0, 0

    if 'resume' in kwargs and kwargs['resume'] is not None:
        print('==> loading checkpoint "{}"'.format(kwargs['resume']))
        checkpoint = torch.load(kwargs['resume'])

        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #
        scheduler.load_state_dict(checkpoint['optimizer'])
        print('==> loaded checkpoint "{}" (epoch {})'.format(kwargs['resume'], epoch))

    writer = tensorboardX.SummaryWriter(save_path)

    no_up = 0
    patience = 5
    for epoch in range(epoch, kwargs['epochs']):

        #if no_up >= patience:
        #    break

        step = epoch * len(dataset['train'])
        print('\n==> epoch {} (starting from step {})'.format(epoch + 1, step + 1))

        # train for one epoch
        if 'scheduler' in kwargs and kwargs['scheduler'] is not None:
            scheduler.step()

        model.train()
        for inputs, targets in tqdm(loaders['train'], desc='train'):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model.forward(inputs)

            # for nets that have multiple outputs such as inception
            if isinstance(outputs, tuple):
                loss = sum((criterion.forward(o, targets) for o in outputs))
            else:
                loss = criterion.forward(outputs, targets)
            # loss = criterion.forward(outputs, targets)

            writer.add_scalar('loss/train', loss.item(), step)
            step += targets.size(0)

            loss.backward()
            optimizer.step()

        # evaluate on testing set
        meters = {}

        model.eval()
        with torch.no_grad():
            for k, meter in kwargs['meters'].items():
                meters[k.format('test')] = meter(**kwargs)

            for inputs, targets in tqdm(loaders['test'], desc='test'):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model.forward(inputs)
                for k, meter in meters.items():
                    meter.add(outputs, targets)

            for k, meter in meters.items():
                meters[k] = meter.value()

        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'meters': meters
        }

        if 'metrics' in kwargs and kwargs['metrics'] is not None:
            if best < meters[kwargs['metrics']]:
                no_up = 0
                best = meters[kwargs['metrics']]
                torch.save(checkpoint, os.path.join(save_path, 'best.pth'))
            else:
                no_up += 1

            meters['{}_best'.format(kwargs['metrics'])] = best

        torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))
        print('==> saved checkpoint "{}"'.format(os.path.join(save_path, 'latest.pth')))

        for k, meter in meters.items():
            print('[{}] = {:2f}'.format(k, meter))
            writer.add_scalar(k, meter, step)

    writer.close()


