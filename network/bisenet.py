import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.bn import InPlaceABNSync as BatchNorm2d
from network.base_models import resnet18

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = BatchNorm2d(out_channels)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.conv2 = ConvBNReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv3 = ConvBNReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv_out = ConvBNReLU(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)

        return x

    def init_weight(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.gap_atten = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn_atten = BatchNorm2d(out_channels, activation='none')
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)

        atten = self.gap_atten(feat)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)

        return out

    def init_weight(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.backbone = resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.refine16 = ConvBNReLU(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.refine32 = ConvBNReLU(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.gap_tail = nn.AdaptiveAvgPool2d(1)
        self.conv_tail = ConvBNReLU(
            in_channels=512,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.init_weight()

    def forward(self, x):
        f8, f16, f32 = self.backbone(x)

        tail = self.gap_tail(f32)
        tail = self.conv_tail(tail)
        tail_up = F.interpolate(tail, size=f32.size()[2:], mode='bilinear', align_corners=True)

        f32_arm = self.arm32(f32)
        f32_sum = f32_arm + tail_up
        f32_up = F.interpolate(f32_sum, size=f16.size()[2:],mode='bilinear', align_corners=True)
        f32_up = self.refine32(f32_up)

        f16_arm = self.arm16(f16)
        f16_sum = f16_arm + f32_up
        f16_up = F.interpolate(f16_sum, size=f8.size()[2:],mode='bilinear', align_corners=True)
        f16_up = self.refine16(f16_up)

        return f16_up, f32_up

    def init_weight(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv = ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.gap_atten = nn.AdaptiveAvgPool2d(1)

        self.conv1_atten = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.relu_atten = nn.ReLU(inplace=True)
        self.conv2_atten = nn.Conv2d(
            in_channels = out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.conv(fcat)

        atten = self.gap_atten(feat)
        atten = self.conv1_atten(atten)
        atten = self.relu_atten(atten)
        atten = self.conv2_atten(atten)
        atten = self.sigmoid_atten(atten)
        atten = torch.mul(feat, atten)
        out = atten + feat

        return out

    def init_weight(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class BiSeNetHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(BiSeNetHead, self).__init__()
        self.conv1 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

    def init_weight(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class BiSeNet(nn.Module):
    def __init__(self, n_classes):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(in_channels=256, out_channels=256)

        self.head_out = BiSeNetHead(in_channels=256, mid_channels=64, out_channels=n_classes)
        self.head_aux16 = BiSeNetHead(in_channels=128, mid_channels=256, out_channels=n_classes)
        self.head_aux32 = BiSeNetHead(in_channels=128, mid_channels=256, out_channels=n_classes)

        self.init_weight()

    def forward(self, x):
        fsp = self.sp(x)
        fcp16, fcp32 = self.cp(x)

        fuse = self.ffm(fsp, fcp16)

        out = self.head_out(fuse)
        aux16 = self.head_aux16(fcp16)
        aux32 = self.head_aux32(fcp32)

        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        aux16 = F.interpolate(aux16, size=x.size()[2:], mode='bilinear', align_corners=True)
        aux32 = F.interpolate(aux32, size=x.size()[2:], mode='bilinear', align_corners=True)

        return out, aux16, aux32

    def init_weight(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def group_weight(self, weight_group, module, lr):
        group_decay = []
        group_no_decay = []

        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, BatchNorm2d):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        weight_group.append(dict(params=group_decay, lr=lr))
        weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
        return weight_group

    def get_params(self, base_lr):
        params = []
        for module in self.children():
            if isinstance(module, ContextPath):
                params = self.group_weight(params, module, base_lr)
            else:
                params = self.group_weight(params, module, base_lr * 10)
        return params

if __name__ == "__main__":
    img = torch.randn(2,3,640,480).cuda()
    net = BiSeNet(19).cuda()
    net.train()
    net.get_params(0.001)
    out ,aux16, aux32= net(img)
    print(aux16.shape)

