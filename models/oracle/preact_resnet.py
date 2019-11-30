from __future__ import absolute_import
import math
import torch.nn as nn


__all__ = ['PreActResNet']

"""
preactivation resnet with bottleneck design.
"""

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(True),
            nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False),
            
            nn.BatchNorm2d(cfg[1]),
            nn.ReLU(True),
            nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False),
            
            nn.BatchNorm2d(cfg[2]),
            nn.ReLU(True),
            nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
            
        )
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.net(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class PreActResNet(nn.Module):
    def __init__(self, num_layers=56, cfg=None, **kwargs):
        super().__init__()
        assert (num_layers - 2) % 9 == 0, 'depth should be 9n+2'

        n = (num_layers - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        )
        
        self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
        self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(cfg[-1], 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x