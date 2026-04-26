import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList()
        # 1x1卷积分支
        self.aspp_blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # 空洞卷积分支（3个不同rate）
        for rate in atrous_rates:
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # ===================== 修复通道数计算错误 =====================
        # 拼接后通道数 = out_channels × (1x1分支 + 空洞卷积分支数 + 全局池化分支)
        # 即：out_channels × (1 + len(atrous_rates) + 1) = out_channels × (len(atrous_rates)+2)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5)  # 新增Dropout，提升泛化能力
        )

    def forward(self, x):
        res = []
        # 1x1 + 空洞卷积分支
        for block in self.aspp_blocks:
            res.append(block(x))
        # 全局池化分支（上采样到原尺寸）
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(
            global_feat, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        res.append(global_feat)
        # 拼接所有分支
        x = torch.cat(res, dim=1)
        # 维度压缩
        x = self.project(x)
        return x


class DeeplabV3Plus(nn.Module):
    def __init__(self, num_classes=6, in_channels=3):  # 默认改为3通道（对齐训练代码）
        super(DeeplabV3Plus, self).__init__()
        # ===================== 核心修改1：移除预训练权重，适配3通道 =====================
        # 加载ResNet50主干，禁用预训练（避免3通道预训练权重与自定义通道冲突）
        backbone = resnet50(weights=None)  # 改为None，避免加载ImageNet预训练权重
        # 动态替换第一层卷积，适配输入通道数
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # 初始化新的卷积层权重（提升训练收敛性）
            nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        # ==============================================================================

        # 提取各阶段特征
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ASPP模块（处理layer4输出的2048维特征）
        self.aspp = ASPP(2048, 256, atrous_rates=[6, 12, 18])

        # 低级特征融合（layer1输出256维→48维）
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 解码头（融合256+48=304维特征）
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),  # 新增Dropout，防止过拟合
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 分割头（256→类别数）
        self.seg_head = nn.Conv2d(256, num_classes, kernel_size=1)

        # 初始化解码器和分割头权重（提升训练收敛速度）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存原始输入尺寸（动态适配任意输入尺寸，不再固定256×256）
        original_size = (x.shape[2], x.shape[3])

        # 主干网络提取多尺度特征
        feat1 = self.layer1(x)  # (B, 256, H/4, W/4) - 低级特征
        feat2 = self.layer2(feat1)  # (B, 512, H/8, W/8)
        feat3 = self.layer3(feat2)  # (B, 1024, H/16, W/16)
        feat4 = self.layer4(feat3)  # (B, 2048, H/32, W/32) - 高级特征

        # ASPP处理高级特征（2048→256）
        aspp_feat = self.aspp(feat4)  # (B, 256, H/32, W/32)

        # 上采样ASPP特征到低级特征尺寸（H/4, W/4）
        aspp_feat = F.interpolate(
            aspp_feat, size=feat1.shape[2:], mode='bilinear', align_corners=False
        )

        # 处理低级特征（256→48）
        low_feat = self.low_level_conv(feat1)  # (B, 48, H/4, W/4)

        # 融合高低级特征（256+48=304）
        fused = torch.cat([aspp_feat, low_feat], dim=1)  # (B, 304, H/4, W/4)

        # 解码融合特征（304→256）
        x = self.decoder(fused)  # (B, 256, H/4, W/4)

        # 分割头预测（256→num_classes）
        x = self.seg_head(x)  # (B, num_classes, H/4, W/4)

        # ===================== 核心修改2：动态上采样到原始尺寸 =====================
        # 不再固定256×256，适配任意输入尺寸（如128/256/512）
        x = F.interpolate(
            x, size=original_size, mode='bilinear', align_corners=False
        )
        return x


# 完整测试代码（验证维度、梯度、鲁棒性）
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型（与训练代码参数完全一致）
    model = DeeplabV3Plus(
        num_classes=6,
        in_channels=3  # 训练代码的RGB 3通道
    ).to(device)

    # 测试1：标准256×256输入（训练代码常用尺寸）
    test_input = torch.randn(8, 3, 256, 256).to(device)  # batch_size=8
    output = model(test_input)
    print("\n=== 标准尺寸测试 ===")
    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")  # 预期：torch.Size([8, 6, 256, 256])
    assert output.shape == (8, 6, 256, 256), "标准尺寸输出不匹配！"

    # 测试2：非标准尺寸鲁棒性（如512×512）
    test_input_large = torch.randn(2, 3, 512, 512).to(device)
    output_large = model(test_input_large)
    print("\n=== 非标准尺寸测试 ===")
    print(f"输入尺寸: {test_input_large.shape}")
    print(f"输出尺寸: {output_large.shape}")  # 预期：torch.Size([2, 6, 512, 512])
    assert output_large.shape == (2, 6, 512, 512), "非标准尺寸输出不匹配！"

    # 测试3：ASPP模块维度验证
    aspp_module = model.aspp
    aspp_input = torch.randn(1, 2048, 8, 8).to(device)  # feat4尺寸（256/32=8）
    aspp_output = aspp_module(aspp_input)
    print("\n=== ASPP模块测试 ===")
    print(f"ASPP输入尺寸: {aspp_input.shape}")
    print(f"ASPP输出尺寸: {aspp_output.shape}")  # 预期：torch.Size([1, 256, 8, 8])
    assert aspp_output.shape == (1, 256, 8, 8), "ASPP模块输出不匹配！"

    # 测试4：梯度传播验证
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    target = torch.randint(0, 6, (8, 256, 256)).to(device)
    loss = loss_fn(output, target)
    loss.backward()

    # 检查关键层梯度
    grad_ok = all([
        model.seg_head.weight.grad is not None,
        model.aspp.project[0].weight.grad is not None,
        model.decoder[0].weight.grad is not None
    ])
    print("\n=== 梯度传播测试 ===")
    print(f"梯度传播: {'✅ 通过' if grad_ok else '❌ 失败'}")
    assert grad_ok, "梯度传播失败！"

    # 测试5：参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== 参数量统计 ===")
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    print("\n✅ 所有测试通过！")