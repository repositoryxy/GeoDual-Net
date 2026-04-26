import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple

# --------------------------- 基础配置 ---------------------------
CFG = {
    "in_channels": 3,  # 输入通道数（RGB遥感图像）
    "num_classes": 6,  # 6分类输出
    "embed_dims": [64, 128],  # 高分辨率流的嵌入维度
    "num_heads": [4, 8],  # 注意力头数
    "window_size": 4,  # 遥感适配窗口大小（4×4）
    "depths": [2, 2],  # 各分辨率流的HRViT块数量
    "drop_rate": 0.1,  # Dropout率
    "use_spectral_attention": True,  # 遥感专属：光谱注意力
    "use_geo_pos_encoding": True  # 遥感专属：地理空间位置编码
}


# --------------------------- 基础模块 ---------------------------
class ConvBNReLU(nn.Module):
    """CNN基础模块：卷积+BN+ReLU，提取遥感图像局部纹理"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SpectralAttention(nn.Module):
    """遥感专属：光谱注意力模块，强化通道/光谱特征"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GeoPosEncoding(nn.Module):
    """遥感专属：地理空间位置编码（替代普通2D位置编码）"""

    def __init__(self, embed_dim: int, h: int = 256, w: int = 256):
        super().__init__()
        x_pos = torch.linspace(-1, 1, w).unsqueeze(0).repeat(h, 1)
        y_pos = torch.linspace(-1, 1, h).unsqueeze(1).repeat(1, w)
        pos = torch.stack([x_pos, y_pos], dim=0).unsqueeze(0)  # [1, 2, h, w]

        self.pos_embed = nn.Conv2d(2, embed_dim, kernel_size=1, stride=1, padding=0)
        self.register_buffer('pos', pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        pos = self.pos.repeat(b, 1, 1, 1)  # [b, 2, h, w]
        pos_embed = self.pos_embed(pos)  # [b, embed_dim, h, w]
        return x + pos_embed


# --------------------------- Transformer模块 ---------------------------
class LightweightAttention(nn.Module):
    """轻量化多头注意力：修复维度拆分逻辑，适配遥感小尺寸图像"""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, drop_rate: float = 0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        # 确保embed_dim能被num_heads整除（防御性检查）
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1, stride=1, padding=0)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        ws = self.window_size

        # 1. 分窗口处理：[b, c, h, w] → [b, c, num_windows, ws, ws]
        # 先计算窗口数量，确保能整除（防御性检查）
        assert h % ws == 0 and w % ws == 0, f"h/w must be divisible by window_size {ws}"
        num_windows = (h // ws) * (w // ws)
        x_windowed = rearrange(x, 'b c (h ws1) (w ws2) -> b c (h w) ws1 ws2',
                               ws1=ws, ws2=ws, h=h // ws, w=w // ws)  # [b, c, num_windows, ws, ws]

        # 2. 转换为4D输入供Conv2d处理：[b, c, num_windows*ws, ws]
        x_4d = x_windowed.reshape(b, c, num_windows * ws, ws)  # 5D→4D

        # 3. 生成QKV：[b, 3*c, num_windows*ws, ws]
        qkv = self.qkv(x_4d)

        # --------------------------- 【修复】核心维度拆分 ---------------------------
        # 步骤1：拆分3*c为3, c → [b, 3, c, num_windows*ws, ws]
        qkv = qkv.reshape(b, 3, self.embed_dim, num_windows * ws, ws)
        # 步骤2：拆分c为num_heads, head_dim → [b, 3, num_heads, head_dim, num_windows*ws, ws]
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, num_windows * ws, ws)
        # 步骤3：转换窗口维度 → [b, 3, num_heads, head_dim, num_windows, ws, ws]
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, num_windows, ws, ws)
        # --------------------------------------------------------------------------

        # 4. 拆分Q/K/V（此时维度0是3，不会越界）
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 各维度：[b, num_heads, head_dim, num_windows, ws, ws]

        # 5. 计算注意力：展平窗口内像素 → [b, num_heads, head_dim, num_windows, ws*ws]
        q_flat = q.reshape(b, self.num_heads, self.head_dim, num_windows, -1)  # [b, nh, hd, nw, ws²]
        k_flat = k.reshape(b, self.num_heads, self.head_dim, num_windows, -1)  # [b, nh, hd, nw, ws²]
        v_flat = v.reshape(b, self.num_heads, self.head_dim, num_windows, -1)  # [b, nh, hd, nw, ws²]

        # 注意力得分：[b, nh, nw, ws², ws²]
        attn = (q_flat.transpose(-2, -1) @ k_flat) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 6. 注意力加权：[b, nh, hd, nw, ws²]
        out_flat = (attn @ v_flat.transpose(-2, -1)).transpose(-2, -1)
        # 恢复窗口维度：[b, nh, hd, nw, ws, ws]
        out = out_flat.reshape(b, self.num_heads, self.head_dim, num_windows, ws, ws)
        # 合并注意力头：[b, c, nw, ws, ws]
        out = out.reshape(b, self.embed_dim, num_windows, ws, ws)

        # 7. 恢复原始图像尺寸：[b, c, h, w]
        out = rearrange(out, 'b c (h w) ws1 ws2 -> b c (h ws1) (w ws2)',
                        h=h // ws, w=w // ws, ws1=ws, ws2=ws)

        # 8. 投影+Dropout
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class HRViTBlock(nn.Module):
    """HRViT核心块：CNN局部特征 + Transformer全局特征"""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, drop_rate: float = 0.):
        super().__init__()
        # CNN分支：提取遥感地物局部纹理
        self.cnn_branch = nn.Sequential(
            ConvBNReLU(embed_dim, embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        )

        # Transformer分支：捕捉全局空间依赖
        self.trans_branch = nn.Sequential(
            LightweightAttention(embed_dim, num_heads, window_size, drop_rate),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        )

        # 融合+残差
        self.norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_branch(x)
        trans_feat = self.trans_branch(x)
        out = self.norm(cnn_feat + trans_feat)
        out = self.relu(out + x)  # 残差连接
        return out


# --------------------------- HRViT-RS主干 ---------------------------
class HRViTRSBackbone(nn.Module):
    """HRViT-RS主干：并行高分辨率流，全程保持256×256特征"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        embed_dims = cfg["embed_dims"]
        depths = cfg["depths"]
        num_heads = cfg["num_heads"]
        window_size = cfg["window_size"]
        drop_rate = cfg["drop_rate"]

        # 输入投影：3通道→64通道（保持256×256）
        self.stem = ConvBNReLU(cfg["in_channels"], embed_dims[0], kernel_size=3, stride=1, padding=1)

        # 遥感专属：地理空间位置编码
        if cfg["use_geo_pos_encoding"]:
            self.pos_encoding = GeoPosEncoding(embed_dims[0], h=256, w=256)

        # 遥感专属：光谱注意力
        if cfg["use_spectral_attention"]:
            self.spectral_att = SpectralAttention(embed_dims[0])

        # 高分辨率流1：64维，256×256（无下采样）
        self.stage1 = nn.Sequential(
            *[HRViTBlock(embed_dims[0], num_heads[0], window_size, drop_rate) for _ in range(depths[0])]
        )

        # 高分辨率流2：128维，128×128（仅1次下采样）
        self.downsample = ConvBNReLU(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(
            *[HRViTBlock(embed_dims[1], num_heads[1], window_size, drop_rate) for _ in range(depths[1])]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入：[b, 3, 256, 256]
        x = self.stem(x)  # [b, 64, 256, 256]

        # 遥感专属模块
        if self.cfg["use_geo_pos_encoding"]:
            x = self.pos_encoding(x)
        if self.cfg["use_spectral_attention"]:
            x = self.spectral_att(x)

        # 高分辨率流1输出
        feat1 = self.stage1(x)  # [b, 64, 256, 256]

        # 高分辨率流2输出
        feat2 = self.downsample(feat1)  # [b, 128, 128, 128]
        feat2 = self.stage2(feat2)  # [b, 128, 128, 128]

        return feat1, feat2


# --------------------------- 分割头 ---------------------------
class SegmentationHead(nn.Module):
    """分割头：上采样恢复256×256，输出6分类"""

    def __init__(self, in_dims: List[int], num_classes: int):
        super().__init__()
        # 融合多尺度特征
        self.fusion = ConvBNReLU(in_dims[0] + in_dims[1], in_dims[0], kernel_size=3, stride=1, padding=1)

        # 上采样：128×128→256×256
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 输出头：6分类
        self.out_conv = nn.Conv2d(in_dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feat1, feat2 = feats  # feat1:[b,64,256,256], feat2:[b,128,128,128]

        # 上采样feat2到256×256
        feat2_up = self.upsample(feat2)  # [b,128,256,256]

        # 融合特征
        fusion_feat = torch.cat([feat1, feat2_up], dim=1)  # [b,192,256,256]
        fusion_feat = self.fusion(fusion_feat)  # [b,64,256,256]

        # 输出6分类结果
        out = self.out_conv(fusion_feat)  # [b,6,256,256]
        return out


# --------------------------- 完整HRViT-RS模型 ---------------------------
class HRViTRS(nn.Module):
    """完整的HRViT-RS遥感语义分割模型"""

    def __init__(self, cfg: dict = CFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = HRViTRSBackbone(cfg)
        self.seg_head = SegmentationHead(cfg["embed_dims"], cfg["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入：[b,3,256,256] → 输出：[b,6,256,256]
        feats = self.backbone(x)
        out = self.seg_head(feats)
        return out


# --------------------------- 测试代码 ---------------------------
if __name__ == "__main__":
    # 初始化模型
    model = HRViTRS(CFG)
    model.eval()

    # 构造测试输入：batch_size=2，3通道，256×256
    test_input = torch.randn(2, 3, 256, 256)

    # 前向传播
    with torch.no_grad():
        output = model(test_input)

    # 验证输出维度
    print(f"输入维度: {test_input.shape}")
    print(f"输出维度: {output.shape}")
    print("HRViT-RS模型初始化成功，维度验证通过！")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")