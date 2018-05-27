import torch.nn as nn
import torch

__all__ = ['I3DNonLocalResNet', 'nonlocal_resnet50', 'nonlocal_resnet101', 'nonlocal_resnet152']


class NonLocalBlock3D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, use_bn=True):
        super(NonLocalBlock3D, self).__init__()

        assert mode in ['embedded_gaussian', 'dot_product']
        self.mode = mode
        if self.mode == 'embedded_gaussian':
            self.operation_function = self._embedded_gaussian
        elif self.mode == 'dot_product':
            self.operation_function = self._dot_product

        self.sub_sample = sub_sample
        self.use_bn = use_bn
        self.in_channels = in_channels
        if inter_channels is None:
            self.inter_channels = in_channels // 2
        else:
            self.inter_channels = inter_channels

        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        if self.sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool3d(kernel_size=2))
            self.phi = nn.Sequential(self.phi, nn.MaxPool3d(kernel_size=2))

        if self.use_bn:
            self.Wz = nn.Sequential(
                nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(self.in_channels,eps=1e-05, momentum=0.9)
            )
        else:
            self.Wz = nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)

        self._init_param()


    def _init_param(self):

        def init_sequential(m):
            if type(m) == nn.Conv3d:
                m.weight.data.normal_(std=0.01)
                m.bias.data.zero_()
            if type(m) == nn.BatchNorm3d:
                m.weight.data.zero_()

        if self.sub_sample:
            self.g.apply(init_sequential)
            self.phi.apply(init_sequential)
        else:
            nn.init.normal(self.g.weight, 0.01)
            nn.init.constant(self.g.bias, 0)
            nn.init.normal(self.phi.weight, 0.01)
            nn.init.constant(self.phi.bias, 0)

        nn.init.normal(self.theta.weight, 0.01)
        nn.init.constant(self.theta.bias, 0)

        if self.use_bn:
            self.Wz.apply(init_sequential)
        else:
            nn.init.normal(self.Wz.weight, 0.01)
            nn.init.constant(self.Wz.bias, 0)


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        output = self.operation_function(x)
        return output


    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # g_x: (b,0.5c,thw)
        g_x = g_x.permute(0, 2, 1) # g_x: (b,thw,0.5c)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) # theta_x: (b,thw,0.5c)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # phi_x: (b,0.5c,thw)

        # apply dot product similarity
        f = torch.matmul(theta_x, phi_x) # f: (b, thw, thw)
        f_div_C = f / f.size(-1) # normalize

        y = torch.matmul(f_div_C, g_x) # y: (b,thw,0.5c)
        y = y.permute(0, 2, 1).contiguous() # y:(b,0.5c,thw)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # y: (b,0.5c,t,h,w)

        z = self.Wz(y) + x
        return z


    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # g_x: (b,0.5c,thw)
        g_x = g_x.permute(0, 2, 1) #g_x: (b,thw,0.5c)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) # theta_x: (b,thw,0.5c)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # phi_x: (b,0.5c,thw)

        # apply embedded gaussian
        f = torch.matmul(theta_x, phi_x) # f: (b, thw, thw)
        f_div_C = nn.functional.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x) # y: (b,thw,0.5c)
        y = y.permute(0, 2, 1).contiguous() # y:(b,0.5c,thw)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # y: (b,0.5c,t,h,w)

        z = self.Wz(y) + x

        return z


def insert_nonlocal_block(input, in_channels, inter_channels=None,
                mode='embedded_gaussian', sub_sample=True, use_bn=True):

    nonlocal_block = NonLocalBlock3D(in_channels, inter_channels=inter_channels, mode=mode,
                                     sub_sample=sub_sample, use_bn=use_bn)
    output = nonlocal_block(input)
    return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, time_kernel=1, space_stride=1, enable_nonlocal=False, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(time_kernel,1,1), padding=((time_kernel-1)//2, 0,0),bias=False) # timepadding: make sure time-dim not reduce
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,space_stride,space_stride),
                               padding=(0,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=(1,1,1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.enable_nonlocal = enable_nonlocal

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

        if self.enable_nonlocal:
            out = insert_nonlocal_block(out, out.size(1))

        return out


class I3DNonLocalResNet(nn.Module):

    def __init__(self, block, layers, nonlocals, frame_num=32, num_classes=1000):
        self.inplanes = 64
        super(I3DNonLocalResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5,7,7), stride=(2,2,2), padding=(2,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.layer1 = self._make_layer_inflat_nonlocal(block, 64, layers[0])
        self.temporalpool = nn.MaxPool3d(kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.layer2 = self._make_layer_inflat_nonlocal(block, 128, layers[1], space_stride=2, nonlocal_index=nonlocals[0])
        self.layer3 = self._make_layer_inflat_nonlocal(block, 256, layers[2], space_stride=2, nonlocal_index=nonlocals[1])
        self.layer4 = self._make_layer_inflat_nonlocal(block, 512, layers[3], space_stride=2)
        self.avgpool = nn.AvgPool3d((frame_num//8,7,7))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer_inflat_nonlocal(self, block, planes, blocks, space_stride=1, nonlocal_index=1000):
        downsample = None
        if space_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1,1), stride=(1,space_stride,space_stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []

        time_kernel = 3 #making I3D(3*1*1)
        enable_nonlocal = False
        layers.append(block(self.inplanes, planes, time_kernel, space_stride, enable_nonlocal, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # making I3D(3*1*1)
            if i % 2 == 0:
                time_kernel = 3
            else:
                time_kernel = 1

            # making nonlocal
            if i % nonlocal_index == nonlocal_index - 1:
                enable_nonlocal = True
            else:
                enable_nonlocal = False

            layers.append(block(self.inplanes, planes, time_kernel, 1, enable_nonlocal, None))

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
        #x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def nonlocal_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DNonLocalResNet(Bottleneck, [3, 4, 6, 3], [2, 2], **kwargs)

    if pretrained:
        import torchvision
        pretrained_model = torchvision.models.resnet50(pretrained=True)
        model = inflat_weights(pretrained_model,model)

    return model


def nonlocal_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DNonLocalResNet(Bottleneck, [3, 4, 23, 3], [2, 7], **kwargs)

    if pretrained:
        import torchvision
        pretrained_model = torchvision.models.resnet101(pretrained=True)
        model = inflat_weights(pretrained_model, model)

    return model


def nonlocal_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DNonLocalResNet(Bottleneck, [3, 8, 36, 3], [4, 12], **kwargs)

    if pretrained:
        import torchvision
        pretrained_model = torchvision.models.resnet152(pretrained=True)
        model = inflat_weights(pretrained_model,model)

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

    resnet_nonlocal = nonlocal_resnet101(pretrained=True)
    print(resnet_nonlocal)
    data = np.ones((1, 3, 32, 224, 224), dtype=np.float32)
    tensor = torch.from_numpy(data)
    inputs = Variable(tensor)
    out = resnet_nonlocal(inputs)
    print(out)