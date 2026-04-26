import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Tuple

# --------------------------- 基础配置 ---------------------------
CFG = {
    "in_channels": 3,  # RGB遥感图像3通道
    "num_classes": 6,  # 6分类输出
    "embed_dim": 96,  # 基础嵌入维度（适配轻量化设计）
    "depths": [2, 2, 6, 2],  # 各阶段ViT块数量（U-Net式4阶段）
    "num_heads": [3, 6, 12, 24],  # 各阶段注意力头数
    "patch_size": 4,  # 遥感图像分块大小（4×4，适配小地物）
    "window_size": 8,  # 窗口注意力大小（8×8，平衡全局/局部）
    "mlp_ratio": 4.,  # MLP隐藏层维度倍率
    "drop_rate": 0.1,  # Dropout率
    "use_geo_pos_encoding": True,  # 遥感专属：地理空间位置编码
    "use_land_prior": True,  # 遥感专属：地物先验特征融合
    "decoder_embed_dim": 64  # 解码器嵌入维度（适配U-Net跳跃连接）
}


# --------------------------- 基础模块 ---------------------------
class ConvBNReLU(nn.Module):
    """CNN基础模块：提取遥感图像局部纹理/边缘特征"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class GeoSpatialPosEncoding(nn.Module):
    """遥感专属：地理空间位置编码（替代普通2D位置编码）"""

    def __init__(self, embed_dim: int, img_size: int = 256, patch_size: int = 4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 生成地理坐标（归一化到[-1,1]，贴合遥感图像的地理坐标系）
        x_pos = torch.linspace(-1, 1, img_size // patch_size)
        y_pos = torch.linspace(-1, 1, img_size // patch_size)
        y, x = torch.meshgrid(y_pos, x_pos, indexing="ij")

        # 编码到嵌入维度
        self.x_embed = nn.Linear(1, embed_dim // 2)
        self.y_embed = nn.Linear(1, embed_dim // 2)

        # 注册为缓冲区（不参与训练）
        self.register_buffer('x_coord', x.reshape(-1, 1))
        self.register_buffer('y_coord', y.reshape(-1, 1))

    def forward(self) -> torch.Tensor:
        # 输出：[1, num_patches, embed_dim]
        x_embed = self.x_embed(self.x_coord)  # [num_patches, embed_dim//2]
        y_embed = self.y_embed(self.y_coord)  # [num_patches, embed_dim//2]
        pos_embed = torch.cat([x_embed, y_embed], dim=-1)  # [num_patches, embed_dim]
        return pos_embed.unsqueeze(0)


class LandPriorFusion(nn.Module):
    """遥感专属：地物先验特征融合
    强化建筑/道路/植被等典型遥感地物的特征表达"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 地物先验权重（6类：建筑、道路、植被、水体、裸地、其他）
        self.land_weights = nn.Parameter(torch.ones(6, in_channels))
        self.conv = ConvBNReLU(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, c, h, w]
        b, c, h, w = x.shape
        # 修复：维度扩展错误，确保weights是[1, c, 1, 1]
        weights = self.softmax(self.land_weights).mean(dim=0, keepdim=True)  # [1, c]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [1, c, 1, 1]
        weighted_feat = x * weights  # [b, c, h, w]（广播正确）
        return self.conv(weighted_feat)


class Attention(nn.Module):
    """轻量化多头注意力：适配遥感小尺寸图像"""

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8, drop_rate: float = 0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, num_patches, dim]
        b, n, c = x.shape

        # 生成QKV
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [b, num_heads, n, head_dim]

        # 窗口注意力（适配遥感图像分块）
        q = q / (self.head_dim ** 0.5)
        attn = (q @ k.transpose(-2, -1))  # [b, num_heads, n, n]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权+投影
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """多层感知机：ViT块的前馈网络"""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop_rate: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SegViTBlock(nn.Module):
    """SegViT核心块：ViT注意力 + CNN局部特征"""

    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float = 4., drop_rate: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, window_size, drop_rate)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop_rate=drop_rate)

        # CNN局部特征补充（遥感图像纹理增强）
        self.cnn_feat = ConvBNReLU(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x: [b, num_patches, dim] → 恢复2D特征图：[b, dim, h, w]
        x_2d = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # CNN局部特征
        cnn_feat = self.cnn_feat(x_2d)
        cnn_feat = rearrange(cnn_feat, 'b c h w -> b (h w) c')

        # ViT注意力
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        # 融合CNN+ViT特征
        x = x + cnn_feat
        return x


# --------------------------- 编码器（ViT+U-Net下采样） ---------------------------
class SegViTEncoder(nn.Module):
    """SegViT编码器：分阶段下采样，提取多尺度遥感特征"""

    def __init__(self, cfg: dict, img_size: int = 256):
        super().__init__()
        self.cfg = cfg
        embed_dim = cfg["embed_dim"]
        depths = cfg["depths"]
        num_heads = cfg["num_heads"]
        patch_size = cfg["patch_size"]
        window_size = cfg["window_size"]
        mlp_ratio = cfg["mlp_ratio"]
        drop_rate = cfg["drop_rate"]

        # 输入投影：3通道→embed_dim（4×4分块，下采样4倍）
        self.patch_embed = nn.Conv2d(cfg["in_channels"], embed_dim, kernel_size=patch_size, stride=patch_size,
                                     padding=0)
        self.num_patches = (img_size // patch_size) ** 2
        self.h, self.w = img_size // patch_size, img_size // patch_size

        # 遥感专属：地理空间位置编码
        if cfg["use_geo_pos_encoding"]:
            self.pos_encoding = GeoSpatialPosEncoding(embed_dim, img_size, patch_size)

        # 修复：将Sequential改为ModuleList，手动循环调用
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # 预计算各阶段通道数（关键：用于解码器匹配）
        self.stage_channels = [embed_dim * (2 ** i) for i in range(len(depths))]

        for i in range(len(depths)):
            # 构建当前阶段的SegViTBlock列表
            stage_blocks = nn.ModuleList([
                SegViTBlock(self.stage_channels[i], num_heads[i], window_size, mlp_ratio, drop_rate)
                for _ in range(depths[i])
            ])
            self.stages.append(stage_blocks)

            # 下采样（最后一阶段不下采样）
            if i < len(depths) - 1:
                downsample = nn.Conv2d(self.stage_channels[i], self.stage_channels[i + 1], kernel_size=2, stride=2,
                                       padding=0)
                self.downsamples.append(downsample)

        # 遥感专属：地物先验融合
        if cfg["use_land_prior"]:
            final_dim = self.stage_channels[-1]
            self.land_fusion = LandPriorFusion(final_dim, final_dim)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[int], List[int], List[int]]:
        # x: [b, 3, 256, 256]
        features = []  # 保存各阶段特征
        hs, ws = [], []  # 保存各阶段特征的高/宽

        # 输入投影+位置编码
        x = self.patch_embed(x)  # [b, 96, 64, 64]
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # [b, 64×64, 96]

        if self.cfg["use_geo_pos_encoding"]:
            pos_embed = self.pos_encoding()  # [1, num_patches, embed_dim]
            x = x + pos_embed

        # 手动循环调用每个阶段的Block
        for i, stage_blocks in enumerate(self.stages):
            # 逐个调用Block
            for block in stage_blocks:
                x = block(x, h, w)

            # 保存当前阶段特征
            x_2d = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            features.append(x_2d)
            hs.append(h)
            ws.append(w)

            # 下采样（最后一阶段除外）
            if i < len(self.downsamples):
                x_2d = self.downsamples[i](x_2d)  # 下采样2倍
                b, c, h, w = x_2d.shape
                x = rearrange(x_2d, 'b c h w -> b (h w) c')

        # 地物先验融合
        if self.cfg["use_land_prior"] and len(features) > 0:
            assert len(features[-1].shape) == 4, f"特征维度错误，期望4D，实际{len(features[-1].shape)}D"
            features[-1] = self.land_fusion(features[-1])

        # 修复：返回各阶段通道数，供解码器使用
        return features, hs, ws, self.stage_channels


# --------------------------- 解码器（U-Net上采样+跳跃连接） ---------------------------
class SegViTDecoder(nn.Module):
    """SegViT解码器：上采样+跳跃连接，恢复256×256分辨率"""

    def __init__(self, cfg: dict, stage_channels: List[int]):
        super().__init__()
        self.cfg = cfg
        self.stage_channels = stage_channels  # 编码器各阶段通道数
        decoder_embed_dim = cfg["decoder_embed_dim"]
        num_classes = cfg["num_classes"]

        # 解码器阶段数（与编码器对称）
        self.num_stages = len(stage_channels)
        self.upconvs = nn.ModuleList()
        self.fusions = nn.ModuleList()

        # 反向构建解码器（从最深层到浅层）
        # 初始输入通道：最深层通道数
        current_dim = stage_channels[-1]
        for i in range(self.num_stages - 1, 0, -1):
            # 上采样：2倍上采样
            upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconvs.append(upconv)

            # 特征融合（跳跃连接）：当前通道 + 上一阶段通道
            fusion_in = current_dim + stage_channels[i - 1]
            fusion = ConvBNReLU(fusion_in, decoder_embed_dim)
            self.fusions.append(fusion)

            # 更新当前通道为解码器嵌入维度
            current_dim = decoder_embed_dim

        # 最终输出头（6分类）
        self.out_conv = nn.Sequential(
            ConvBNReLU(decoder_embed_dim, decoder_embed_dim),
            nn.Conv2d(decoder_embed_dim, num_classes, kernel_size=1, stride=1, padding=0)
        )

        # 最终上采样（恢复到256×256）
        self.final_upsample = nn.Upsample(scale_factor=cfg["patch_size"], mode='bilinear', align_corners=True)

    def forward(self, features: List[torch.Tensor], hs: List[int], ws: List[int]) -> torch.Tensor:
        # 从最深层特征开始解码
        x = features[-1]

        # 逐阶段上采样+融合
        for i in range(len(self.upconvs)):
            # 上采样到对应尺寸
            x = self.upconvs[i](x)
            # 获取对应的跳跃连接特征
            skip_feat = features[-(i + 2)]
            # 确保尺寸匹配
            x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=True)
            # 拼接特征
            x = torch.cat([x, skip_feat], dim=1)
            # 特征融合
            x = self.fusions[i](x)

        # 最终上采样到256×256
        x = self.final_upsample(x)
        # 输出6分类结果
        out = self.out_conv(x)
        return out


# --------------------------- 完整SegViT-RS模型 ---------------------------
class SegViTRS(nn.Module):
    """完整的SegViT-RS遥感语义分割模型（ViT+U-Net混合架构）"""

    def __init__(self, cfg: dict = CFG, img_size: int = 256):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size

        # 初始化编码器
        self.encoder = SegViTEncoder(cfg, img_size)
        # 修复：解码器接收编码器的阶段通道数，确保通道匹配
        self.decoder = SegViTDecoder(cfg, self.encoder.stage_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入：[b, 3, 256, 256]
        features, hs, ws, _ = self.encoder(x)
        # 解码+输出：[b, 6, 256, 256]
        out = self.decoder(features, hs, ws)
        return out


# --------------------------- 测试代码 ---------------------------
if __name__ == "__main__":
    # 初始化模型
    model = SegViTRS(CFG, img_size=256)
    model.eval()

    # 构造测试输入：batch_size=2，3通道，256×256
    test_input = torch.randn(2, 3, 256, 256)

    # 前向传播
    with torch.no_grad():
        output = model(test_input)

    # 验证输出维度（应输出[2,6,256,256]）
    print(f"输入维度: {test_input.shape}")
    print(f"输出维度: {output.shape}")
    print("SegViT-RS模型初始化成功，维度验证通过！")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")