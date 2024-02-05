import torch.nn as nn
import torch
import math
# from torchsummary import summary
from torchinfo import summary


import torch.nn.functional as F

class ReparameterizedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReparameterizedConvolution, self).__init__()

        # Large convolutional layer
        self.large_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

        # Medium convolutional layer
        self.medium_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Small convolutional layer
        self.small_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0)

    def forward(self, x):
        # Forward pass through all convolutional layers
        large_conv_output = self.large_conv(x)
        medium_conv_output = self.medium_conv(x)
        small_conv_output = self.small_conv(x)

        # Get the spatial dimensions of the large convolutional layer
        _, _, h, w = large_conv_output.size()

        # Calculate the starting indices for cropping (for medium_conv)
        start_h_medium = (large_conv_output.size(2) - medium_conv_output.size(2)) // 2
        start_w_medium = (large_conv_output.size(3) - medium_conv_output.size(3)) // 2

        # Crop the central part of the large convolutional layer for medium_conv
        medium_conv_output = medium_conv_output[:, :, start_h_medium:start_h_medium + large_conv_output.size(2), start_w_medium:start_w_medium + large_conv_output.size(3)]

        # Calculate the starting indices for cropping (for small_conv)
        start_h_small = (large_conv_output.size(2) - small_conv_output.size(2)) // 2
        start_w_small = (large_conv_output.size(3) - small_conv_output.size(3)) // 2

        # Crop the central part of the large convolutional layer for small_conv
        small_conv_output = small_conv_output[:, :, start_h_small:start_h_small + large_conv_output.size(2), start_w_small:start_w_small + large_conv_output.size(3)]

        # Sum the outputs to perform the reparameterization
        output = large_conv_output + medium_conv_output + small_conv_output

        return output







def ResLCnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResLCnet([1, 2, 1, 1], **kwargs)
    return model


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # 逐通道卷积：groups=in_channels=out_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        # 逐点卷积：普通1x1卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        # 7x7 convolutional layer
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=11, padding=5)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        # 3x3 convolutional layer
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out0 = self.conv0(x)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        # You can concatenate or combine the results in any way you need.
        # For example, you can use torch.cat to concatenate them along the channel dimension.
        out = torch.cat([out2, out3], dim=1)
        out = torch.cat([out1, out], dim=1)
        out = torch.cat([out0, out], dim=1)

        return out

class ResLCnet(nn.Module):
    """
    block: A sub module
    """

    def __init__(self, layers, num_classes=1000):
        super(ResLCnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),            
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SeparableConv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # InceptionModule(64, 16),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            ReparameterizedConvolution(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self.make_stack(64, layers[0], stride=2, ker=15),

            self.make_stack(128, layers[1], stride=2, ker=11),

            self.make_stack(256, layers[2], stride=2, ker=9),

            self.make_stack(512, layers[3], stride=2, ker=3)
        )

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024, num_classes)
        # initialize parameters
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride, ker):
        layers = []
        downsample = nn.Sequential(
            SeparableConv2d(planes, planes * 2, 3, 2, 1),
            nn.BatchNorm2d(planes * 2),
            nn.Conv2d(in_channels=planes * 2, out_channels=planes * 2, kernel_size=1, stride=1),
        )

        layers.append(Bottleneck(planes, planes * 2, stride, downsample, ker=ker))
        for i in range(1, blocks):
            layers.append(Bottleneck(planes * 2, planes * 2, stride=1, downsample=None, ker=ker))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, ker=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.gn = nn.GroupNorm(planes, planes)
        self.conv2 = SeparableConv2d(planes, planes, ker, stride, (ker - 1) // 2)

        self.convx = nn.Conv2d(planes, planes, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.gelu = nn.GELU()
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv1(x)
        out = self.gn(out)
        out = self.conv2(out)
        out = self.convx(out)
        out += residual
        add = out
        
        out = self.bn(out)
        out = self.conv3(out)
        out = self.gelu(out)
        out = self.conv4(out)
        out += add

        out += residual
        out = self.relu(out)

        return out

net = ResLCnet50(num_classes=71)
summary(net.features, input_size=(1,3, 224, 224))  # 打印网络结构
