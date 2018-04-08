import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, time_kernel=1, space_stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(time_kernel,1,1), padding=((time_kernel-1)/2, 0,0),bias=False) # timepadding: make sure time-dim not reduce
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,space_stride,space_stride),
                               padding=(0,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=(1,1,1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class I3DResNet(nn.Module):

    def __init__(self, block, layers, frame_num=32, num_classes=1000):
        self.inplanes = 64
        super(I3DResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5,7,7), stride=(2,2,2), padding=(2,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.layer1 = self._make_layer_inflat(block, 64, layers[0])
        self.temporalpool = nn.MaxPool3d(kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.layer2 = self._make_layer_inflat(block, 128, layers[1], space_stride=2)
        self.layer3 = self._make_layer_inflat(block, 256, layers[2], space_stride=2)
        self.layer4 = self._make_layer_inflat(block, 512, layers[3], space_stride=2)
        self.avgpool = nn.AvgPool3d((frame_num/8,7,7))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer_inflat(self, block, planes, blocks, space_stride=1):
        downsample = None
        if space_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1,1), stride=(1,space_stride,space_stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        time_kernel = 3 #making I3D(3*1*1)
        layers.append(block(self.inplanes, planes, time_kernel, space_stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i % 2 == 0:
                time_kernel = 3
            else:
                time_kernel = 1
            layers.append(block(self.inplanes, planes, time_kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.temporalpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        import torchvision
        pretrained_model = torchvision.models.resnet101(pretrained=True)
        model = inflat_weights(pretrained_model, model)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


def inflat_weights(model_2d, model_3d):
    pretrained_dict_2d = model_2d.state_dict()
    model_dict_3d = model_3d.state_dict()
    for key,weight_2d in pretrained_dict_2d.items():
        if key in model_dict_3d:
            if 'conv' in key:
                time_kernel_size = model_dict_3d[key].shape[2]
                if 'weight' in key:
                    weight_3d = weight_2d.unsqueeze(2).repeat(1,1,time_kernel_size,1,1)
                    weight_3d = weight_3d / time_kernel_size
                    model_dict_3d[key] = weight_3d
                elif 'bias' in key:
                    model_dict_3d[key] = weight_2d
            elif 'bn' in key:
                    model_dict_3d[key] = weight_2d
            elif 'fc' in key:
                if 'weight' in key:
                    time_kernel_size = model_dict_3d[key].shape[1] / weight_2d.shape[1]
                    weight_3d = weight_2d.repeat(1, time_kernel_size)
                    weight_3d = weight_3d / time_kernel_size
                    model_dict_3d[key] = weight_3d
                elif 'bias' in key:
                    model_dict_3d[key] = weight_2d
            elif 'downsample' in key:
                if '0.weight' in key:
                    time_kernel_size = model_dict_3d[key].shape[2]
                    weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_kernel_size, 1, 1)
                    weight_3d = weight_3d / time_kernel_size
                    model_dict_3d[key] = weight_3d
                else:
                    model_dict_3d[key] = weight_2d

    model_3d.load_state_dict(model_dict_3d)
    return model_3d


if __name__ == '__main__':
    import torchvision
    import numpy as np
    import torch
    from torch.autograd import Variable

    resnet = torchvision.models.resnet101(pretrained=True)
    resnet_i3d = resnet101(pretrained=True)

    data = np.ones((1, 3, 224, 224), dtype=np.float32)
    tensor = torch.from_numpy(data)
    inputs = Variable(tensor)
    out1 = resnet(inputs)
    print out1

    data2 = np.ones((1, 3, 32, 224, 224), dtype=np.float32)
    tensor2 = torch.from_numpy(data2)
    inputs2 = Variable(tensor2)
    out2 = resnet_i3d(inputs2)
    print out2