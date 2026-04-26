import torch
import torch.nn as nn
import torch.nn.functional as F


# 双卷积块（Conv + ReLU + Conv + ReLU）
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),  # 移除bias（配合BN更稳定）
            nn.BatchNorm2d(out_channels),  # 新增BN层（提升训练稳定性，对齐主流UNet实现）
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 下采样块（MaxPool + ConvBlock，通道数翻倍）
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 下采样：尺寸减半
            ConvBlock(in_channels, out_channels)  # 卷积：通道数翻倍
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# 上采样块（转置Conv + 拼接跳跃连接 + ConvBlock）
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # 转置卷积：上采样（尺寸翻倍）+ 通道数减半（in_channels → in_channels//2）
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False
        )
        # 拼接后通道数 = 转置卷积输出（in_channels//2） + 跳跃连接通道数（out_channels）
        self.conv = ConvBlock(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        # 1. 上采样（转置卷积）
        x1 = self.up(x1)

        # 2. 处理尺寸不匹配（鲁棒性优化：支持奇数尺寸）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 对称padding，避免尺寸偏差
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        # 3. 通道拼接（跳跃连接：x2在前，x1在后，符合UNet标准）
        x = torch.cat([x2, x1], dim=1)

        # 4. 双卷积降维
        return self.conv(x)


# 完整UNet模型（兼容训练代码的features配置，鲁棒性增强）
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, features=[64, 128, 256, 256]):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.n_levels = len(features)  # 下采样层数（4层）
        self.bottleneck_out = features[-1] * 2  # 瓶颈层输出通道数（256×2=512，适配训练代码的features）

        # 初始卷积（输入通道→第一个特征通道）
        self.in_conv = ConvBlock(in_channels, features[0])

        # 下采样模块（兼容任意长度的features列表）
        self.downs = nn.ModuleList()
        for i in range(self.n_levels - 1):
            self.downs.append(DownBlock(features[i], features[i + 1]))

        # 瓶颈层（最后一个特征通道→2倍通道数）
        self.bottleneck = ConvBlock(features[-1], self.bottleneck_out)

        # 上采样模块（核心修复：兼容features=[64,128,256,256]）
        self.ups = nn.ModuleList()
        # 上采样输入通道数：[瓶颈层输出, 256, 128] → 对应训练代码的features
        up_in_channels = [self.bottleneck_out] + features[1:-1][::-1]
        # 上采样输出通道数：[256, 128, 64]
        up_out_channels = features[:-1][::-1]
        # 安全校验：确保上采样层数匹配
        assert len(up_in_channels) == len(up_out_channels), \
            f"上采样通道数不匹配！in:{len(up_in_channels)}, out:{len(up_out_channels)}"

        for in_feat, out_feat in zip(up_in_channels, up_out_channels):
            self.ups.append(UpBlock(in_feat, out_feat))

        # 输出层（最后一个特征通道→类别数）
        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # 1. 下采样 + 保存跳跃连接
        skip_connections = []
        x = self.in_conv(x)  # 3→64，尺寸256×256
        for down in self.downs:
            skip_connections.append(x)  # 保存：64、128、256
            x = down(x)  # 下采样：64→128→256→256（适配训练代码的features）

        # 2. 瓶颈层：256→512，尺寸32×32
        x = self.bottleneck(x)

        # 3. 上采样 + 拼接跳跃连接（反转跳跃连接列表）
        skip_connections = skip_connections[::-1]  # [256, 128, 64]
        # 安全校验：跳跃连接数匹配上采样层数
        assert len(skip_connections) == len(self.ups), \
            f"跳跃连接数不匹配！skip:{len(skip_connections)}, ups:{len(self.ups)}"

        for idx, up in enumerate(self.ups):
            x = up(x, skip_connections[idx])  # 上采样+拼接+卷积

        # 4. 输出：64→6，尺寸256×256
        logits = self.out_conv(x)
        return logits


# 测试代码（完全对齐训练逻辑，验证正确性）
if __name__ == "__main__":
    # 初始化模型（与训练代码参数完全一致）
    model = UNet(
        in_channels=3,  # 训练代码的RGB 3通道
        num_classes=6,  # 类别数
        features=[64, 128, 256, 256]  # 训练代码中的features配置
    )

    # 模拟训练输入（batch_size=8，3通道，256×256）
    x = torch.randn(8, 3, 256, 256)
    # 前向传播
    output = model(x)

    # 打印关键信息验证
    print("=" * 50)
    print(f"输入尺寸：{x.shape}")  # 预期：torch.Size([8, 3, 256, 256])
    print(f"输出尺寸：{output.shape}")  # 预期：torch.Size([8, 6, 256, 256])
    print("✅ 尺寸匹配测试通过！")

    # 计算参数量（验证模型复杂度）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量：{total_params / 1e6:.2f}M")
    print(f"可训练参数量：{trainable_params / 1e6:.2f}M")

    # 梯度测试（验证反向传播）
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, torch.randint(0, 6, (8, 256, 256)))
    loss.backward()
    # 检查关键层梯度
    grad_ok = model.out_conv.weight.grad is not None
    print(f"\n梯度传播测试：{'✅ 通过' if grad_ok else '❌ 失败'}")
    print("=" * 50)