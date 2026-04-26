import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===================== Basic utility functions =====================
def count_params(model):
    """Count model parameters (in millions)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


class SwinConfig:
    """Swin Transformer configuration class (adapted for decoder scale)"""

    def __init__(self):
        self.embed_dim = 64  # Initial decoder dimension
        self.depths = [2, 2, 2, 2]  # Number of Swin blocks per decoder layer
        self.num_heads = [2, 4, 8, 16]  # Number of attention heads per decoder layer
        self.window_size = 4  # Decoder window size
        self.mlp_ratio = 4.0
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0


# ===================== Basic modules =====================
class ConvBNAct(nn.Module):
    """Basic convolution block: Conv + BN + ReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------- ResNet-16 encoder modules --------------------
class ResNet16Block(nn.Module):
    """ResNet-16 basic residual block"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, stride, 1)
        self.conv2 = ConvBNAct(out_ch, out_ch, 3, 1, 1)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = ConvBNAct(in_ch, out_ch, 1, stride, 0)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual  # non-inplace addition
        return F.relu(out)


class ResNet16Encoder(nn.Module):
    """ResNet-16 single-branch encoder"""

    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 64, 7, 2, 3)  # 256→128
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # 128→64

        # 4 residual layers
        self.layer1 = ResNet16Block(64, 64, stride=1)  # 64×64
        self.layer2 = ResNet16Block(64, 128, stride=2)  # 32×32
        self.layer3 = ResNet16Block(128, 256, stride=2)  # 16×16
        self.layer4 = ResNet16Block(256, 512, stride=2)  # 8×8

    def forward(self, x):
        x = self.stem(x)  # (B,64,128,128)
        x = self.maxpool(x)  # (B,64,64,64)

        feat1 = self.layer1(x)  # (B,64,64,64)
        feat2 = self.layer2(feat1)  # (B,128,32,32)
        feat3 = self.layer3(feat2)  # (B,256,16,16)
        feat4 = self.layer4(feat3)  # (B,512,8,8)

        skip_features = [feat1, feat2, feat3, feat4]
        return feat4, skip_features


# -------------------- Self-attention skip connection module --------------------
class SelfAttentionSkip(nn.Module):
    """Self-attention enhanced skip connection module"""

    def __init__(self, in_ch):
        super().__init__()
        self.ch = in_ch
        self.query_conv = nn.Conv2d(in_ch, in_ch // 8, 1, 1, 0)
        self.key_conv = nn.Conv2d(in_ch, in_ch // 8, 1, 1, 0)
        self.value_conv = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Generate Q/K/V
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, H * W)
        value = self.value_conv(x).view(B, -1, H * W)

        # Self-attention computation
        attn = self.softmax(torch.bmm(query, key) / math.sqrt(self.ch // 8))
        out = torch.bmm(value, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        # Attention weighted fusion
        out = self.gamma * out + x
        return out


# -------------------- Swin Transformer auxiliary decoder modules --------------------
def window_partition(x, window_size):
    """Divide into windows"""
    B, H, W, C = x.shape
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    H_pad, W_pad = H + pad_b, W + pad_r

    x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows, (pad_r, pad_b)


def window_reverse(windows, window_size, H, W, pad):
    """Merge windows"""
    pad_r, pad_b = pad
    H_pad = H + pad_b
    W_pad = W + pad_r
    B = int(windows.shape[0] / (H_pad * W_pad / window_size / window_size))
    x = windows.view(B, H_pad // window_size, W_pad // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H_pad, W_pad, -1)
    if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :].contiguous()
    return x


class WindowAttention(nn.Module):
    """Swin window attention (adapted for decoder)"""

    def __init__(self, dim, num_heads, window_size, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

        # Initialization
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # QKV decomposition
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Mask handling
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # Output
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinDecoderBlock(nn.Module):
    """Swin Transformer auxiliary decoder block (ultimate fixed version)"""

    def __init__(self, in_dim, out_dim, num_heads, window_size, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim  # Explicitly specify output dimension, not automatically computed
        self.window_size = window_size

        # 1. Upsample + dimension reduction layer (force match output dimension)
        self.upsample = nn.ConvTranspose2d(self.in_dim, self.out_dim, kernel_size=2, stride=2, padding=0)

        # 2. LayerNorm (normalization dimension set to out_dim)
        self.norm1 = nn.LayerNorm(self.out_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.out_dim, eps=1e-6)

        # 3. Window attention (input dimension is out_dim)
        self.attn = WindowAttention(
            dim=self.out_dim,
            num_heads=num_heads,
            window_size=window_size,
            attn_drop=attn_drop
        )

        # 4. MLP layer (adapted for out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim, int(self.out_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(self.out_dim * mlp_ratio), self.out_dim),
            nn.Dropout(drop)
        )

        # 5. Skip connection adapter layer
        self.skip_adapter = nn.Conv2d(self.out_dim, self.out_dim, 1, 1, 0, bias=False)

    def forward(self, x, skip_feat=None):
        """
        x: (B, in_dim, H, W) input features
        skip_feat: (B, C, H', W') skip connection features
        Returns: (B, out_dim, 2H, 2W) upsampled features
        """
        B, C, H, W = x.shape
        assert C == self.in_dim, f"Input channel {C} does not match expected {self.in_dim}"

        # 1. Upsample + dimension reduction (force output specified dimension)
        x = self.upsample(x)  # (B, out_dim, 2H, 2W)
        B, C_up, H_up, W_up = x.shape
        assert C_up == self.out_dim, f"After upsampling channel {C_up} does not match expected {self.out_dim}"

        # 2. Fuse skip connection features
        shortcut = x  # Residual connection
        if skip_feat is not None:
            # Align spatial size
            if x.shape[2:] != skip_feat.shape[2:]:
                skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            # Align channels
            if skip_feat.shape[1] != C_up:
                skip_feat = nn.Conv2d(skip_feat.shape[1], C_up, 1, 1, 0).to(x.device)(skip_feat)
            # Residual fusion
            x = x + self.skip_adapter(skip_feat)

        # 3. Swin Transformer computation
        # (B, C_up, H_up, W_up) → (B, H_up, W_up, C_up)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Normalization + window attention
        x_norm = self.norm1(x)
        x_windows, pad = window_partition(x_norm, self.window_size)  # (B*num_win, win, win, C_up)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C_up)  # (B*num_win, win², C_up)
        attn_out = self.attn(x_windows)  # (B*num_win, win², C_up)

        # Merge windows
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C_up)
        x_attn = window_reverse(attn_out, self.window_size, H_up, W_up, pad)  # (B, H_up, W_up, C_up)

        # Residual connection
        x = shortcut.permute(0, 2, 3, 1).contiguous() + x_attn

        # MLP layer
        x = x + self.mlp(self.norm2(x))

        # Restore shape (B, H_up, W_up, C_up) → (B, C_up, H_up, W_up)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


# -------------------- Main decoder block (traditional convolution decoder) --------------------
class MainDecoderBlock(nn.Module):
    """Main decoder block (traditional convolution decoding)"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.skip_ch = skip_ch
        self.out_ch = out_ch
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2, 0)
        self.conv1 = ConvBNAct(in_ch // 2 + skip_ch, out_ch, 3, 1, 1)
        self.conv2 = ConvBNAct(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, skip_feat=None):
        x = self.upsample(x)
        if skip_feat is not None and self.skip_ch > 0:
            if x.shape[2:] != skip_feat.shape[2:]:
                x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# -------------------- Feature merging module --------------------
class FeatureMerge(nn.Module):
    """Main/auxiliary decoder output merging module (dynamic channel configuration)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, main_feat, aux_feat):
        # Align spatial size
        if main_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=main_feat.shape[2:], mode='bilinear', align_corners=False)
        # Align channels (fallback)
        if main_feat.shape[1] != aux_feat.shape[1]:
            aux_feat = nn.Conv2d(aux_feat.shape[1], main_feat.shape[1], 1, 1, 0).to(main_feat.device)(aux_feat)
        # Concatenate features
        fused = torch.cat([main_feat, aux_feat], dim=1)
        # Convolution fusion
        return self.conv(fused)


# ===================== Improved main model =====================
class Res16_DualDecoder_SegModel(nn.Module):
    """
    Ultimate fixed version model:
    - Left: ResNet-16 single-branch encoder
    - Right: Main decoder (traditional convolution) + Auxiliary decoder (Swin Transformer)
    - Skip connections: self-attention enhanced
    - Dual decoders at each layer receive the same input, outputs are merged and passed to the next layer
    - Adapted for Potsdam 6-class remote sensing dataset (output 256×256)
    """

    def __init__(self, num_classes=6, swin_config=None):
        super().__init__()
        self.num_classes = num_classes
        self.swin_config = swin_config or SwinConfig()

        # -------------------- Encoder --------------------
        self.encoder = ResNet16Encoder()

        # -------------------- Self-attention skip connections --------------------
        self.attention_skips = nn.ModuleList([
            SelfAttentionSkip(64),  # layer1 feature enhancement
            SelfAttentionSkip(128),  # layer2 feature enhancement
            SelfAttentionSkip(256),  # layer3 feature enhancement
            SelfAttentionSkip(512)   # layer4 feature enhancement
        ])

        # -------------------- Decoder configuration (exact input/output channels per layer) --------------------
        self.decoder_layers = [
            # Layer 4→3: input 512 → main output 256 + aux output 256 → concatenated 512 → merged output 256
            {
                "main": {"in": 512, "skip": 256, "out": 256},
                "aux": {"in": 512, "out": 256, "heads": 2},
                "merge": {"in": 256 + 256, "out": 256}
            },
            # Layer 3→2: input 256 → main output 128 + aux output 128 → concatenated 256 → merged output 128
            {
                "main": {"in": 256, "skip": 128, "out": 128},
                "aux": {"in": 256, "out": 128, "heads": 4},
                "merge": {"in": 128 + 128, "out": 128}
            },
            # Layer 2→1: input 128 → main output 64 + aux output 64 → concatenated 128 → merged output 64
            {
                "main": {"in": 128, "skip": 64, "out": 64},
                "aux": {"in": 128, "out": 64, "heads": 8},
                "merge": {"in": 64 + 64, "out": 64}
            },
            # Layer 1→output: input 64 → main output 64 + aux output 64 → concatenated 128 → merged output 64
            {
                "main": {"in": 64, "skip": 0, "out": 64},
                "aux": {"in": 64, "out": 64, "heads": 16},
                "merge": {"in": 64 + 64, "out": 64}
            }
        ]

        # -------------------- Main decoder (traditional convolution) --------------------
        self.main_decoder = nn.ModuleList([
            MainDecoderBlock(
                in_ch=layer["main"]["in"],
                skip_ch=layer["main"]["skip"],
                out_ch=layer["main"]["out"]
            ) for layer in self.decoder_layers
        ])

        # -------------------- Auxiliary decoder (Swin Transformer) --------------------
        self.aux_decoder = nn.ModuleList([
            SwinDecoderBlock(
                in_dim=layer["aux"]["in"],
                out_dim=layer["aux"]["out"],
                num_heads=layer["aux"]["heads"],
                window_size=self.swin_config.window_size,
                mlp_ratio=self.swin_config.mlp_ratio,
                drop=self.swin_config.drop_rate,
                attn_drop=self.swin_config.attn_drop_rate
            ) for layer in self.decoder_layers
        ])

        # -------------------- Feature merging modules (per layer) --------------------
        self.merge_modules = nn.ModuleList([
            FeatureMerge(
                in_channels=layer["merge"]["in"],
                out_channels=layer["merge"]["out"]
            ) for layer in self.decoder_layers
        ])

        # -------------------- Output layer (core modification: changed stride from 4 to 2) --------------------
        self.final_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, output_padding=0)  # 128×128→256×256
        self.seg_head = nn.Sequential(
            ConvBNAct(64, 32, 3, 1, 1),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(32, num_classes, 1, 1, 0)
        )

        # Initialize weights
        self.apply(self._init_weights)
        print(f"Improved model initialization complete | Trainable parameters: {count_params(self):.2f}M")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward propagation logic"""
        # 1. Encoder forward
        encoder_feat, skip_features = self.encoder(x)  # encoder_feat: (B,512,8,8)

        # 2. Self-attention enhanced skip connection features
        enhanced_skips = [self.attention_skips[i](skip_features[i]) for i in range(4)]

        # 3. Decoder forward (from deepest to shallowest)
        dec_input = encoder_feat  # Initial input: last encoder layer features
        skip_indices = [2, 1, 0, None]  # Skip feature index for each layer

        for i in range(4):
            # Get current layer configuration and skip features
            layer_cfg = self.decoder_layers[i]
            skip_feat = enhanced_skips[skip_indices[i]] if skip_indices[i] is not None else None

            # Main/auxiliary decoder receive same input (both dec_input)
            main_feat = self.main_decoder[i](dec_input, skip_feat)
            aux_feat = self.aux_decoder[i](dec_input, skip_feat)

            # Merge outputs as input for next layer
            dec_input = self.merge_modules[i](main_feat, aux_feat)

        # 4. Final upsampling + classification output
        out = self.final_upsample(dec_input)  # 128×128→256×256
        out = self.seg_head(out)  # (B,6,256,256)

        return out


# ===================== Test code =====================
if __name__ == "__main__":
    # Initialize model
    model = Res16_DualDecoder_SegModel(num_classes=6)

    # Test input (Potsdam dataset: 256×256×3)
    x = torch.randn(1, 3, 256, 256)

    # Forward pass verification
    with torch.no_grad():
        out = model(x)

    # Verify output size
    print(f"Input size: {x.shape}")
    print(f"Output size: {out.shape}")  # Expected: (1,6,256,256)
    print(f"Model trainable parameters: {count_params(model):.2f}M")