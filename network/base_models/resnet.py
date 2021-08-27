import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo

from modules.bn import InPlaceABNSync as BatchNorm2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

def conv1x1(in_channels, out_channel, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channel,
        kernel_size=1,
        stride=stride,
        bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            in_channels=in_channels,
            out_channels=channels,
            stride=stride
        )
        self.bn1 = BatchNorm2d(channels)
        self.conv2 = conv3x3(
            in_channels=channels,
            out_channels=channels
        )
        self.bn2 = BatchNorm2d(channels, activation='none')
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_channels != channels * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(
                    in_channels=in_channels,
                    out_channel=channels*self.expansion,
                    stride=stride
                ),
                BatchNorm2d(
                    channels * self.expansion,
                    activation='none'
                )
            )

    def forward(self, x):
        shortcut = x

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = residual+ shortcut
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_channels, channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channel=channels,
        )
        self.bn1 = BatchNorm2d(channels)

        self.conv2 = conv3x3(
            in_channels=channels,
            out_channels=channels,
            stride=stride
        )
        self.bn2 = BatchNorm2d(channels)

        self.conv3 = conv1x1(
            in_channels=channels,
            out_channel=channels * self.expansion
        )
        self.bn3 = BatchNorm2d(channels * self.expansion, activation='none')
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_channels != channels * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(
                    in_channels=in_channels,
                    out_channel=channels*self.expansion,
                    stride=stride
                ),
                BatchNorm2d(
                    channels * self.expansion,
                    activation='none'
                )
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, stride=1, pretrained=True, name='resnet18'):
        super(ResNet, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = 64
        self.layer1 = self._make_layer(
            block=block,
            channels=64,
            num=layers[0],
            stride=1
        )
        self.layer2 = self._make_layer(
            block=block,
            channels=128,
            num=layers[1],
            stride=2
        )
        self.layer3 = self._make_layer(
            block=block,
            channels=256,
            num=layers[2],
            stride=2
        )
        self.layer4 = self._make_layer(
            block=block,
            channels=512,
            num=layers[3],
            stride=2
        )

        self.init_weight()


    def _make_layer(self, block, channels, num, stride=1):

        layers = []
        layers.append(
            block(
                in_channels=self.in_channels,
                channels=channels,
                stride=stride
            )
        )
        self.in_channels = channels * block.expansion

        for i in range(1, num):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    channels=channels,
                    stride=1
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f8 = self.layer2(x)
        f16 = self.layer3(f8)
        f32 = self.layer4(f16)

        return f8, f16, f32

    def init_weight(self):
        state_dict = modelzoo.load_url(model_urls[self.name])
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2],name='resnet18')
    return model

def resnet101(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],name='resnet101')
    return model

if __name__ == '__main__':
    # resnet18 = models.resnet18(pretrained=True)

    #for name, module in resnet18.named_modules():
    #   print(name, module)
    #resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    # state_dict = modelzoo.load_url(resnet18_url)
    #for k, v in state_dict.items():
    #    print(k

    x = torch.randn(1,3,1024,1024)

    net = resnet18()

    f8, f16, f32 = net(x)
    print(f16.shape)

    x = torch.randn(16, 3, 224, 224)
    out = net(x)

    #for name, module in net.named_modules():
       # print(name, module)




