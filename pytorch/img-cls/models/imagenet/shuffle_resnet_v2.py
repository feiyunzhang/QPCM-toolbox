#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import torch
import torch.nn as nn
from .shufflenetv2_slim import *

__all__ = ['shuffle_resnet50_v2', 'shuffle_resnet101_v2', 'shuffle_resnet152_v2']


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

class BasicResBlock(nn.Module):
    def __init__(self, name, in_channels, out_channels, stride, dilation):
        super(BasicResBlock, self).__init__()
        self.g_name = name
        self.in_channels = in_channels
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        channels = out_channels//2
        if stride == 1:
            #assert in_channels == out_channels
            self.conv = nn.Sequential(
                conv_bn_relu(name + '/conv1', channels, channels, 1),
                conv_bn(name + '/conv2',
                    channels, channels, 3, stride=stride,
                    dilation=dilation, padding=dilation, groups=channels),
                conv_bn(name + '/conv3', channels, channels, 1),
            )
        else:
            self.conv = nn.Sequential(
                conv_bn_relu(name + '/conv1', in_channels, channels, 1),
                conv_bn(name + '/conv2',
                    channels, channels, 3, stride=stride,
                    dilation=dilation, padding=dilation, groups=channels),
                conv_bn(name + '/conv3', channels, channels, 1),
            )
            self.conv0 = nn.Sequential(
                conv_bn(name + '/conv4',
                    in_channels, in_channels, 3, stride=stride,
                    dilation=dilation, padding=dilation, groups=in_channels),
                conv_bn_relu(name + '/conv5', in_channels, channels, 1),
            )
            self.downsample = nn.Sequential(
                conv_bn(name + '/downsample', in_channels, channels, 1, stride=stride), 
            )
        self.shuffle = channel_shuffle(name + '/shuffle', 2)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            res_x2 = self.conv(x2) + x2
            res_x2 = self.relu(res_x2)
            x = torch.cat((x1, res_x2), 1)
        else:
            res_x0 = self.conv0(x) + self.downsample(x)
            res_x0 = self.relu(res_x0)
            x = torch.cat((res_x0, self.conv(x)), 1)
        return self.shuffle(x)

class BasicResBlock_Con(nn.Module):
    def __init__(self, name, in_channels, out_channels, stride, dilation):
        super(BasicResBlock_Con, self).__init__()
        self.g_name = name
        self.in_channels = in_channels
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        channels = out_channels//2
        self.conv = nn.Sequential(
            conv_bn_relu(name + '/conv1', in_channels, channels, 1),
            conv_bn(name + '/conv2',
                channels, channels, 3, stride=stride,
                dilation=dilation, padding=dilation, groups=channels),
            conv_bn(name + '/conv3', channels, channels, 1),
            )
        self.conv0 = nn.Sequential(
            conv_bn(name + '/conv4',
                in_channels, in_channels, 3, stride=stride,
                dilation=dilation, padding=dilation, groups=in_channels),
            conv_bn_relu(name + '/conv5', in_channels, channels, 1),
            )
        self.shuffle = channel_shuffle(name + '/shuffle', 2)

    def forward(self, x):
        x = torch.cat((self.conv0(x), self.conv(x)), 1)
        return self.shuffle(x)

class ShuffleResNetV2(nn.Module):
    '''
    width_facter_2.0: (244, 488, 976, 1952)
    '''
    def __init__(self, num_classes, layer_nums):
        super(ShuffleResNetV2, self).__init__()
        layer_config = {
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3),
        }
        depth_config = layer_config[layer_nums]
        self.num_classes = num_classes
        in_channels = 64

        # outputs, stride, dilation, blocks, type
        self.network_config = [
            #g_name('data/bn', nn.BatchNorm2d(3)),
            conv_bn_relu('stage1/conv', 3, in_channels, 3, 2, 1),
            g_nam('stage1/pool', nn.MaxPool2d(3, 2, 0, ceil_mode=True)),
            (244, 1, 1, depth_config[0], 'b'), #stage2
            (488, 2, 1, depth_config[1], 'b'), # stage3
            (976, 2, 1, depth_config[2], 'b'), # stage4
            (1952, 2, 1, depth_config[3], 'b'), # stage5
            conv_bn_relu('conv6', 1952, 2048, 1),
            g_name('pool', nn.AvgPool2d(7, 1)),
            g_name('fc', nn.Conv2d(2048, self.num_classes, 1)),
        ]
        self.network = []
        for i, config in enumerate(self.network_config):
            if isinstance(config, nn.Module):
                self.network.append(config)
                continue
            out_channels, stride, dilation, num_blocks, stage_type = config
            stage_prefix = 'stage_{}'.format(i - 1)
            if out_channels == 244:
                blocks = [BasicResBlock_Con(stage_prefix + '_1', in_channels,
                    out_channels, stride, dilation)]
            else:
                blocks = [BasicResBlock(stage_prefix + '_1', in_channels,
                    out_channels, stride, dilation)] 
            for i in range(1, num_blocks):
                blocks.append(BasicResBlock(stage_prefix + '_{}'.format(i + 1),
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
    def forward(self, x):
        x = self.network(x)
        return x.reshape(x.shape[0], -1)

def shuffle_resnet50_v2(num_classes=1000):
    model = ShuffleResNetV2(num_classes=num_classes, layer_nums=50)
    return model

def shuffle_resnet101_v2(num_classes=1000):
    model = ShuffleResNetV2(num_classes=num_classes, layer_nums=101)
    return model

def shuffle_resnet152_v2(num_classes=1000):
    model = ShuffleResNetV2(num_classes=num_classes, layer_nums=152)
    return model