import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# ==================== 标准 PPM 金字塔池化模块 ====================
class PPM(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), channels=512):
        super().__init__()
        self.blocks = nn.ModuleList()
        for ps in pool_sizes:
            self.blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        size = x.shape[2:]
        out = [x]
        for block in self.blocks:
            out.append(F.interpolate(block(x), size=size, mode='bilinear', align_corners=False))
        return torch.cat(out, dim=1)


# ==================== 标准 FPN ====================
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, inputs):
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False
            )
        return [conv(x) for conv, x in zip(self.fpn_convs, laterals)]


# ==================== 标准 UperNet (无Bug版) ====================
class UperNet(nn.Module):
    def __init__(self, num_classes=6, in_channels=3):
        super().__init__()

        # Backbone
        backbone = resnet50(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # FPN
        self.fpn = FPN([256, 512, 1024, 2048], out_channels=256)

        # PPM
        self.ppm = PPM(in_channels=2048, pool_sizes=(1, 2, 3, 6), channels=512)

        # 解码器
        self.decode = nn.Sequential(
            nn.Conv2d(256 * 4 + 2048 + 512 * 4, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # FPN 融合
        fpn_outs = self.fpn([f1, f2, f3, f4])
        fpn_ups = [F.interpolate(f, (h, w), mode='bilinear', align_corners=False) for f in fpn_outs]
        fpn_cat = torch.cat(fpn_ups, dim=1)

        # PPM
        ppm_out = self.ppm(f4)
        ppm_up = F.interpolate(ppm_out, (h, w), mode='bilinear', align_corners=False)

        # 最终融合
        combined = torch.cat([fpn_cat, ppm_up], dim=1)
        out = self.decode(combined)
        return out


# ==================== 测试（必加 model.eval()） ====================
if __name__ == "__main__":
    model = UperNet(num_classes=6, in_channels=3)
    model.eval()  # 关闭 BatchNorm 训练模式，解决报错！
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)