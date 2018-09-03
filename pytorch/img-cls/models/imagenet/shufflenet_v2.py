#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import torch
import torch.nn as nn

#import .shufflenetv2_slim
from .shufflenetv2_slim import *

__all__ = ['shufflenet_v2']


class BasicBlock(nn.Module):

    def __init__(self, name, in_channels, out_channels, stride, dilation):
        super(BasicBlock, self).__init__()
        self.g_name = name
        self.in_channels = in_channels
        self.stride = stride
        channels = out_channels//2
        if stride == 1:
            assert in_channels == out_channels
            self.conv = nn.Sequential(
                conv_bn_relu(name + '/conv1', channels, channels, 1),
                conv_bn(name + '/conv2', 
                    channels, channels, 3, stride=stride, 
                    dilation=dilation, padding=dilation, groups=channels),
                conv_bn_relu(name + '/conv3', channels, channels, 1),
            )
        else:
            self.conv = nn.Sequential(
                conv_bn_relu(name + '/conv1', in_channels, channels, 1),
                conv_bn(name + '/conv2', 
                    channels, channels, 3, stride=stride, 
                    dilation=dilation, padding=dilation, groups=channels),
                conv_bn_relu(name + '/conv3', channels, channels, 1),
            )
            self.conv0 = nn.Sequential(
                conv_bn(name + '/conv4', 
                    in_channels, in_channels, 3, stride=stride, 
                    dilation=dilation, padding=dilation, groups=in_channels),
                conv_bn_relu(name + '/conv5', in_channels, channels, 1),
            )
        self.shuffle = channel_shuffle(name + '/shuffle', 2)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            x = torch.cat((x1, self.conv(x2)), 1)
        else:
            x = torch.cat((self.conv0(x), self.conv(x)), 1)
        return self.shuffle(x)


class ShuffleNetV2(nn.Module):

    def __init__(self, num_classes, width_multiplier):
        super(ShuffleNetV2, self).__init__()
        width_config = {
            0.25: (24, 48, 96, 512),
            0.33: (32, 64, 128, 512),
            0.5: (48, 96, 192, 1024),
            1.0: (116, 232, 464, 1024),
            1.5: (176, 352, 704, 1024),
            2.0: (244, 488, 976, 2048),
        }
        width_config = width_config[width_multiplier]
        self.num_classes = num_classes
        in_channels = 24

        # outputs, stride, dilation, blocks, type
        self.network_config = [
            g_name('data/bn', nn.BatchNorm2d(3)),
            conv_bn_relu('stage1/conv', 3, in_channels, 3, 2, 1),
            # g_name('stage1/pool', nn.MaxPool2d(3, 2, 1)),
            g_name('stage1/pool', nn.MaxPool2d(3, 2, 0, ceil_mode=True)),
            (width_config[0], 2, 1, 4, 'b'),
            (width_config[1], 2, 1, 8, 'b'), # x16
            (width_config[2], 2, 1, 4, 'b'), # x32
            conv_bn_relu('conv5', width_config[2], width_config[3], 1),
            g_name('pool', nn.AvgPool2d(7, 1)),
            g_name('fc', nn.Conv2d(width_config[3], self.num_classes, 1)),
        ]
        self.network = []
        for i, config in enumerate(self.network_config):
            if isinstance(config, nn.Module):
                self.network.append(config)
                continue
            out_channels, stride, dilation, num_blocks, stage_type = config
            stage_prefix = 'stage_{}'.format(i - 1)
            blocks = [BasicBlock(stage_prefix + '_1', in_channels, 
                out_channels, stride, dilation)]
            for i in range(1, num_blocks):
                blocks.append(BasicBlock(stage_prefix + '_{}'.format(i + 1), 
                    out_channels, out_channels, 1, dilation))
            self.network += [nn.Sequential(*blocks)]

            in_channels = out_channels
        self.network = nn.Sequential(*self.network)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

        #for name, m in self.named_modules():
            #if any(map(lambda x: isinstance(m, x), [nn.Linear, nn.Conv1d, nn.Conv2d])):
                #nn.init.kaiming_uniform_(m.weight, mode='fan_in')
                #if m.bias is not None:
                    #nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.network(x)
        return x.reshape(x.shape[0], -1)

def shufflenet_v2(num_classes=1000, widen_factor=2.0):
    model = ShuffleNetV2(num_classes=num_classes, width_multiplier=widen_factor)
    return model
