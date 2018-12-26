from __future__ import division

"""
Creates a ResNet Model as defined in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2015). 
Deep Residual Learning for Image Recognition. 
arXiv preprint arXiv:1512.03385.
import from https://github.com/facebook/fb.resnet.torch
Copyright (c) Yang Lu, 2017
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from ..dropblock.dropblock import DropBlock2D

__all__ = ['resnet18_dropblock']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dropblock, teacher_model, stage, index, blocks, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, dropblock, teacher_model, stage, index, blocks, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockDrop(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dropblock, teacher_model, stage, index, blocks, stride=1, downsample=None):
        super(BasicBlockDrop, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.dropblock = dropblock
        self.teacher_model = teacher_model
        self.index = index
        self.stage = stage
        self.blocks = blocks
    def forward(self, x):
        if self.training:
            residual = x[0]
            t_conv1 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].conv1(x[1])')
            t_bn1 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].bn1(t_conv1)')
            t_relu1 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].relu(t_bn1)')
              
            t_conv2 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].conv2(t_relu1)')
            out = self.conv1(x[0])
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropblock((self.conv2(out), t_conv2))
            out = self.bn2(out)
            if self.downsample is not None:
                residual = self.downsample(x[0])

            #residual = self.dropblock((residual,x[1]))   
            
            t_out = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '](x[1])')
        else:
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropblock(self.conv2(out))
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        if self.training and self.index != (self.blocks -1):
            return (out, t_out)
        else:
            return out


class BottleneckDrop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, dropblock, teacher_model, stage, index, blocks, stride=1, downsample=None):
        super(BottleneckDrop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropblock = dropblock
        self.teacher_model = teacher_model
        self.stage = stage
        self.index = index 
        self.blocks = blocks

    def forward(self, x):
        if self.training:
            residual = x[0]
            t_conv1 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].conv1(x[1])')
            t_bn1 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].bn1(t_conv1)')
            t_relu1 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].relu(t_bn1)')

            t_conv2 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].conv2(t_relu1)')
            t_bn2 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].bn2(t_conv2')
            t_relu2 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].relu2(t_bn2)')
            t_conv3 = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '].conv3(t_relu2)')

            out = self.dropblock(self.conv1(x[0]), t_conv1)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.dropblock(self.conv2(out), t_conv2)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.dropblock(self.conv3(out), t_conv3)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)
        
            residual = self.dropblock((residual,x[1]))
            t_out = eval('self.teacher_model.layer' + str(self.stage) + '[' + str(self.index) + '](x[1])')
        else:
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x) 
        out += residual
        out = self.relu(out)

        if self.training and self.index != (self.blocks -1):
            return (out, t_out)
        else:
            return out


class ResDropNet(nn.Module):
    def __init__(self, bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000, drop_prob=0.1, block_size=7, teacher_model=None):
        """ Constructor
        Args:
            layers: config of layers, e.g., (3, 4, 23, 3)
            num_classes: number of classes
        """
        super(ResDropNet, self).__init__()
        if bottleneck:
            block = Bottleneck
            block_drop = BottleneckDrop
        else:
            block = BasicBlock
            block_drop = BasicBlockDrop

        self.inplanes = baseWidth  # default 64

        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
        self.teacher_model = teacher_model
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        
        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, baseWidth, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth)
        else:
            self.conv1 = nn.Conv2d(3, baseWidth // 2, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(baseWidth // 2)
            self.conv2 = nn.Conv2d(baseWidth // 2, baseWidth // 2, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(baseWidth // 2)
            self.conv3 = nn.Conv2d(baseWidth // 2, baseWidth, 3, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(baseWidth)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, baseWidth, layers[0])
        self.layer2 = self._make_layer(block, baseWidth * 2, layers[1], 2)
        self.layer3 = self._make_layer(block_drop, baseWidth * 4, layers[2], 2, stage=3)
        self.layer4 = self._make_layer(block_drop, baseWidth * 8, layers[3], 2, stage=4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(baseWidth * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, stage=None):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNet
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.dropblock, self.teacher_model, stage, 0, blocks, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.dropblock, self.teacher_model, stage, i, blocks))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = x
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        if self.training:
            y = self.teacher_model.conv1(y)
            y = self.teacher_model.bn1(y)
            y = self.teacher_model.relu(y)
            y = self.teacher_model.conv2(y)
            y = self.teacher_model.bn2(y)
            y = self.teacher_model.relu(y)
            y = self.teacher_model.conv3(y)
            y = self.teacher_model.bn3(y)
            y = self.teacher_model.relu(y)
            y = self.teacher_model.maxpool(y)
            y = self.teacher_model.layer1(y)
            y1 = self.teacher_model.layer2(y)
            y2 = self.teacher_model.layer3(y1)
            y3 = self.teacher_model.layer4(y2)
            y3 = self.teacher_model.avgpool(y3)
            y3 = y3.view(y3.size(0), -1)
            y3 = self.teacher_model.fc(y3)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.training:
            x = self.layer3((x, y1))
            x = self.layer4((x, y2))
        else:
            x = self.layer3(x)
            x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training:
            return (x,y3)
        else:
            return x


def resnet(bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000, drop_prob=0.1, block_size=7, teacher_model=None):
    """
    Construct ResNet.
    (2, 2, 2, 2) for resnet18	# bottleneck=False
    (2, 2, 2, 2) for resnet26
    (3, 4, 6, 3) for resnet34	# bottleneck=False
    (3, 4, 6, 3) for resnet50
    (3, 4, 23, 3) for resnet101
    (3, 8, 36, 3) for resnet152
    note: if you use head7x7=False, the actual depth of resnet will increase by 2 layers.
    """
    model = ResDropNet(bottleneck=bottleneck, baseWidth=baseWidth, head7x7=head7x7, layers=layers, num_classes=num_classes, drop_prob=drop_prob, block_size=block_size, teacher_model = teacher_model)
    return model


def resnet18_dropblock(drop_prob=0.2, block_size=7, teacher_model = None):
    model = ResDropNet(bottleneck=False, baseWidth=64, head7x7=False, layers=(2, 2, 2, 2), num_classes=1000, drop_prob=drop_prob, block_size=block_size, teacher_model = teacher_model)
    return model




