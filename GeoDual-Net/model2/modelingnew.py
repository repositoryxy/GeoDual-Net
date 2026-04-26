import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import config as configs
from resnet_skip_new import TransResNetV2
from model_resnet import *

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Convert HWIO to OIHW if needed"""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(0.2)  # Enhanced regularization
        self.proj_dropout = Dropout(0.2)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.3)  # Enhanced regularization

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        img_size = _pair(img_size)
        self.hybrid_model = TransResNetV2(config, block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)

    def forward(self, input):
        x, features = self.hybrid_model(input)
        return x, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SemanticAlign(nn.Module):
    """Semantic alignment module: channel mapping + lightweight attention"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(out_ch // 8, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_ch // 8, 1), out_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        w = self.att(x)
        return x * w


class ChannelSpatialAttention(nn.Module):
    """Channel + spatial attention fusion enhancement"""
    def __init__(self, in_ch, reduction=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        x = x * ca
        sa = self.spatial_att(x)
        x = x * sa
        return x


class ASPP(nn.Module):
    """Multi-scale context capture module"""
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # Enhanced regularization
        )

    def forward(self, x):
        res = []
        for b in self.branches:
            res.append(b(x))
        gp = self.global_pool(x)
        gp = torch.nn.functional.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(gp)
        x = torch.cat(res, dim=1)
        x = self.project(x)
        return x


class ImprovedDecoderBlock(nn.Module):
    """Improved decoder block (with Dropout regularization)"""
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.align = SemanticAlign(skip_channels, out_channels) if skip_channels > 0 else None

        # Clear layer structure to ensure index consistency
        self.conv_block = nn.Sequential(
            # First convolution
            nn.Conv2d(
                in_channels=out_channels + (out_channels if skip_channels > 0 else 0),
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # New Dropout (index 3)

            # Second convolution
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)   # New Dropout (index 7)
        )

        self.fusion_att = ChannelSpatialAttention(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            skip = self.align(skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        x = self.fusion_att(x)
        return x


class ImprovedSegmentationHead(nn.Module):
    """Segmentation output head"""
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2 if in_channels >= 32 else in_channels, 
                      kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(in_channels // 2 if in_channels >= 32 else in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2 if in_channels >= 32 else in_channels, out_channels, kernel_size=1)
        )
        self.upsample = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=False) if upsampling > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class ImprovedDecoderCup(nn.Module):
    """Decoder body"""
    def __init__(self, config, use_aspp=True, aspp_out_ch=512):
        super().__init__()
        self.config = config
        decoder_channels = config.decoder_channels
        head_channels = 1024
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = list(decoder_channels)

        self.use_aspp = use_aspp
        if use_aspp:
            self.aspp = ASPP(head_channels, aspp_out_ch, rates=(1, 6, 12))
            self.aspp_proj = nn.Sequential(
                nn.Conv2d(aspp_out_ch, head_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True)
            )

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels.copy()
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        self.blocks = nn.ModuleList([
            ImprovedDecoderBlock(in_ch, out_ch, sk_ch, use_batchnorm=True)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ])

        final_ch = 16
        self.final_project = nn.Sequential(
            nn.Conv2d(out_channels[-1], final_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_ch),
            nn.ReLU(inplace=True)
        )

        self.boundary_head = nn.Sequential(
            nn.Conv2d(final_ch, final_ch//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_ch//2, 1, kernel_size=1)
        )

        self.conv_more = Conv2dReLU(head_channels, head_channels, kernel_size=3, padding=1, use_batchnorm=True)

    def forward(self, x, features=None):
        x = self.conv_more(x)
        if self.use_aspp:
            x_aspp = self.aspp(x)
            x = self.aspp_proj(x_aspp)

        for i, block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = block(x, skip=skip)

        x = self.final_project(x)
        boundary_logits = self.boundary_head(x)
        return x, boundary_logits


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=256, num_classes=6, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size)
        self.decoder = ImprovedDecoderCup(config)
        self.segmentation_head = ImprovedSegmentationHead(
            in_channels=16,
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        input_size = x.shape[2:]
        x, features = self.transformer(x)
        x, boundary = self.decoder(x, features)
        logits = self.segmentation_head(x)

        if logits.shape[2:] != input_size:
            logits = torch.nn.functional.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return logits


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}