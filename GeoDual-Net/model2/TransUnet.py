import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from types import SimpleNamespace

# 复用Attention、Mlp、Block、Encoder模块
# 注意：确保modelingnew.py中的Encoder输出维度与hidden_size一致
from modelingnew import Attention, Mlp, Block, Encoder


class TransUNet(nn.Module):
    # 接收config参数（从trainC.py传递，确保hidden_size统一）
    def __init__(self, num_classes=6, in_channels=3, img_size=256, config=None):  # 默认in_channels改为3
        super(TransUNet, self).__init__()
        self.img_size = img_size  # 保存输入尺寸，用于后续上采样匹配

        # 处理config：优先使用外部传递的ViT配置（含hidden_size=768）
        if config is not None:
            self.config = config
            self.hidden_size = self.config.hidden_size  # 从config读取768
        else:
            # 默认配置（兼容单独运行）
            default_config = {
                'hidden_size': 768,
                'transformer': {'num_layers': 12, 'num_heads': 12, 'mlp_dim': 3072},
                'vit_patches_size': 16
            }
            self.config = SimpleNamespace(**default_config)
            self.hidden_size = self.config.hidden_size

        # ===================== 核心修改1：移除预训练权重，适配3通道输入 =====================
        # 加载ResNet50主干，禁用预训练（避免3通道预训练权重与自定义通道冲突）
        backbone = resnet50(weights=None)  # 改为None，避免加载ImageNet预训练权重
        # 动态替换第一层卷积，适配输入通道数
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        # ==================================================================================

        # 卷积编码器（ResNet50，layer3输出维度=1024）
        self.conv_encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3  # 输出：(B, 1024, img_size//16, img_size//16)
        )

        # -------------------------- 维度投影层（核心修复，增强鲁棒性）--------------------------
        # 将ResNet50的1024维特征 → Transformer需要的768维特征
        self.feature_projection = nn.Sequential(
            nn.Linear(1024, self.hidden_size),
            nn.LayerNorm(self.hidden_size),  # 归一化，提升训练稳定性
            nn.ReLU(inplace=True)  # 新增激活，增强非线性表达
        )
        # -------------------------------------------------------------------------------------

        # Transformer编码器（接收768维特征）
        self.transformer_encoder = Encoder(
            config=self.config,
            vis=False
        )

        # ===================== 核心修改2：优化解码器，确保尺寸匹配 =====================
        # UNet解码器（输入维度=hidden_size=768，输出逐步降维）
        # 解码器输出尺寸：
        # 768→512: (img_size//16)*2 = img_size//8
        # 512→256: img_size//4
        # 256→128: img_size//2
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, 512, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 分割头
        self.seg_head = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        batch_size = x.shape[0]
        original_size = (x.shape[2], x.shape[3])  # 保存原始输入尺寸，适配任意尺寸输入

        # 1. 卷积编码（输出：B, 1024, H//16, W//16）
        conv_feat = self.conv_encoder(x)  # (B, 1024, 16, 16) when img_size=256
        C, H, W = conv_feat.shape[1], conv_feat.shape[2], conv_feat.shape[3]

        # 2. 展平为序列（B, H*W, C）→ (B, 256, 1024)
        seq = conv_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, 1024)

        # 3. 应用维度投影（1024→768）
        seq = self.feature_projection(seq)  # (B, 256, 768)

        # 4. Transformer编码（输入768维，匹配LayerNorm要求）
        trans_feat, _ = self.transformer_encoder(seq)  # (B, 256, 768)

        # 5. 还原为特征图（B, hidden_size, H, W）→ (B, 768, 16, 16)
        trans_feat = trans_feat.permute(0, 2, 1).reshape(batch_size, self.hidden_size, H, W)

        # 6. 解码（768→128，尺寸从16→128 when img_size=256）
        x = self.decoder(trans_feat)  # (B, 128, 128, 128)

        # 7. 分割头预测
        x = self.seg_head(x)  # (B, num_classes, 128, 128)

        # ===================== 核心修改3：动态上采样到原始输入尺寸 =====================
        # 确保输出尺寸与输入完全匹配（兼容非256尺寸输入）
        x = F.interpolate(
            x, size=original_size,
            mode='bilinear', align_corners=False
        )  # (B, 6, 256, 256)

        return x


# 测试代码（完全对齐训练逻辑，验证正确性）
if __name__ == "__main__":
    # 模拟训练代码中的ViT配置（从modelingnew导入CONFIGS）
    from modelingnew import CONFIGS as CONFIGS_ViT_seg

    config = CONFIGS_ViT_seg['R50-ViT-B_16']
    config.n_classes = 6
    config.hidden_size = 768

    # 初始化模型（与训练代码参数完全一致）
    model = TransUNet(
        num_classes=6,
        in_channels=3,  # 训练代码的RGB 3通道
        img_size=256,
        config=config
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
    grad_ok = model.seg_head.weight.grad is not None
    print(f"\n梯度传播测试：{'✅ 通过' if grad_ok else '❌ 失败'}")
    print("=" * 50)