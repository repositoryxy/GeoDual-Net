import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import json
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ======================== 1. Core Configuration ========================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset
DATASET_NAME = 'Vai_256'
ROOT_PATH = r'/root/autodl-tmp/ST-Unet/datasets/Vaihingen/npz_data_RGB_improved'

# Weight path (change to your own pth file)
MODEL_WEIGHT_PATH = r"/root/autodl-tmp/ST-Unet/ComResult/Res16_DualDecoder_networks/Res16_DualDecoder_Vai_256_256/swin_embed64_win4/iter30k_epo150_bs8_lr0.01_s1234/RGBepoch_150.pth"

# Training config json (mean and std come from this)
CONFIG_PATH = r"/root/autodl-tmp/ST-Unet/datasets/Vaihingen/config.json"

# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_CLASSES = 6
IN_CHANNELS = 3
VIS_NUM = 399
DEBUG = True

# Output save path
OUTPUT_DIR = r'/root/autodl-tmp/ST-Unet/TestResults/Res16_DualDecoder'

# Classes
CLASS_NAMES = [
    'Low shrub', 'Impervious surfaces', 'Building',
    'Background', 'Vegetation', 'Vehicle'
]
VALID_CLASSES = [0, 1, 2, 3, 4, 5]
CLASS_COLORS = [
    (0, 255, 255), (255, 255, 255), (0, 0, 255),
    (255, 0, 0), (0, 255, 0), (255, 255, 0)
]

# ======================== 2. Data Loading ========================
class SegTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, img_size=256, transform=None):
        self.root_path = root_path
        self.img_size = img_size
        self.transform = transform

        self.file_paths = []
        for f in os.listdir(root_path):
            if f.endswith('.npz'):
                self.file_paths.append(os.path.join(root_path, f))

        print(f"📁 Loaded {len(self.file_paths)} test samples")

        if DEBUG:
            all_labels = []
            sample_num = min(500, len(self.file_paths))
            for i in range(sample_num):
                data = np.load(self.file_paths[i])
                all_labels.extend(np.unique(data['label']))
            all_labels = np.unique(all_labels)
            print(f"🔍 Label range of first 500 samples: {all_labels}")
            print(f"🔍 Included classes: {[CLASS_NAMES[int(l)] for l in all_labels]}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz_path = self.file_paths[idx]
        file_name = os.path.basename(npz_path).replace('.npz', '')

        data = np.load(npz_path)
        img = data['image']
        mask = data['label']

        if self.transform is not None:
            img_trans = img.transpose(1, 2, 0)
            img_trans = self.transform(img_trans)
            img = img_trans.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask, file_name

# ======================== 3. Evaluation Metrics ========================
class SegMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        for pred, target in zip(preds, targets):
            mask = (target >= 0) & (target < self.num_classes)
            pred_valid = pred[mask]
            target_valid = target[mask]

            if len(pred_valid) > 0 and len(target_valid) > 0:
                self.confusion_matrix += confusion_matrix(
                    target_valid, pred_valid,
                    labels=list(range(self.num_classes))
                )

    def compute(self):
        cm = self.confusion_matrix.copy()
        results = {}

        iou = []
        precision = []
        recall = []
        f1 = []
        for cls in range(self.num_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp

            iou_cls = tp / (tp + fp + fn + 1e-8)
            precision_cls = tp / (tp + fp + 1e-8)
            recall_cls = tp / (tp + fn + 1e-8)
            f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + 1e-8)

            iou.append(iou_cls)
            precision.append(precision_cls)
            recall.append(recall_cls)
            f1.append(f1_cls)

        valid_mask = np.array([cls in VALID_CLASSES for cls in range(self.num_classes)])
        results['mIoU'] = float(np.mean(np.array(iou)[valid_mask]))
        results['mPrecision'] = float(np.mean(np.array(precision)[valid_mask]))
        results['mRecall'] = float(np.mean(np.array(recall)[valid_mask]))
        results['mF1'] = float(np.mean(np.array(f1)[valid_mask]))

        total_tp = np.diag(cm).sum()
        total_samples = cm.sum()
        results['OA'] = float(total_tp / (total_samples + 1e-8))

        results['per_class_iou'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(iou)}
        results['per_class_precision'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(precision)}
        results['per_class_recall'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(recall)}
        results['per_class_f1'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(f1)}

        return results

# ======================== 4. Visualization ========================
def vis_result(img, mask, pred, file_name, save_path, mean, std):
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * std + mean

    if mean.max() > 1:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    def mask2color(mask_data):
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.cpu().numpy()
        h, w = mask_data.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in range(len(CLASS_COLORS)):
            color_img[mask_data == cls] = CLASS_COLORS[cls]
        return color_img

    mask_color = mask2color(mask)
    pred_color = mask2color(pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Model Prediction')
    axes[2].axis('off')

    plt.suptitle(file_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ======================== 🔥 Your provided model: Res16_DualDecoder_SegModel (fully pasted) ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

class SwinConfig:
    def __init__(self):
        self.embed_dim = 64
        self.depths = [2, 2, 2, 2]
        self.num_heads = [2, 4, 8, 16]
        self.window_size = 4
        self.mlp_ratio = 4.0
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResNet16Block(nn.Module):
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
        out = out + residual
        return F.relu(out)

class ResNet16Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = ResNet16Block(64, 64, stride=1)
        self.layer2 = ResNet16Block(64, 128, stride=2)
        self.layer3 = ResNet16Block(128, 256, stride=2)
        self.layer4 = ResNet16Block(256, 512, stride=2)
    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        return feat4, [feat1, feat2, feat3, feat4]

class SelfAttentionSkip(nn.Module):
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
        query = self.query_conv(x).view(B, -1, H*W).permute(0,2,1)
        key = self.key_conv(x).view(B, -1, H*W)
        value = self.value_conv(x).view(B, -1, H*W)
        attn = self.softmax(torch.bmm(query, key) / math.sqrt(self.ch//8))
        out = torch.bmm(value, attn.permute(0,2,1))
        out = out.view(B,C,H,W)
        return self.gamma * out + x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    x = F.pad(x, (0,0,0,pad_r,0,pad_b))
    Hp, Wp = H+pad_b, W+pad_r
    x = x.view(B, Hp//window_size, window_size, Wp//window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows, (pad_r, pad_b)

def window_reverse(windows, window_size, H, W, pad):
    pad_r, pad_b = pad
    Hp, Wp = H+pad_b, W+pad_r
    B = int(windows.shape[0] / (Hp*Wp / window_size / window_size))
    x = windows.view(B, Hp//window_size, Wp//window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*window_size-1)*(2*window_size-1), num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = torch.flatten(coords,1)
        rel_coords = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        rel_coords = rel_coords.permute(1,2,0).contiguous()
        rel_coords[:,:,0] += window_size-1
        rel_coords[:,:,1] += window_size-1
        rel_coords[:,:,0] *= 2*window_size-1
        self.rel_index = rel_coords.sum(-1)
        self.register_buffer("relative_position_index", self.rel_index)
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1))
        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size**2, self.window_size**2, -1)
        attn = attn + rel_bias.permute(2,0,1).unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_,N,C)
        x = self.proj(x)
        return self.proj_drop(x)

class SwinDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, window_size, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.window_size = window_size
        self.upsample = nn.ConvTranspose2d(in_dim, out_dim, 2,2,0)
        self.norm1 = nn.LayerNorm(out_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(out_dim, eps=1e-6)
        self.attn = WindowAttention(out_dim, num_heads, window_size, attn_drop)
        self.mlp = nn.Sequential(nn.Linear(out_dim, int(out_dim*mlp_ratio)), nn.GELU(), nn.Dropout(drop), nn.Linear(int(out_dim*mlp_ratio), out_dim), nn.Dropout(drop))
        self.skip_adapter = nn.Conv2d(out_dim, out_dim,1,1,0,bias=False)
    def forward(self, x, skip_feat=None):
        x = self.upsample(x)
        B, C, H, W = x.shape
        shortcut = x
        if skip_feat is not None:
            if x.shape[2:] != skip_feat.shape[2:]:
                skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            if skip_feat.shape[1] != C:
                skip_feat = nn.Conv2d(skip_feat.shape[1], C,1,1,0).to(x.device)(skip_feat)
            x = x + self.skip_adapter(skip_feat)
        x = x.permute(0,2,3,1)
        x_norm = self.norm1(x)
        wins, pad = window_partition(x_norm, self.window_size)
        wins = wins.view(-1, self.window_size**2, C)
        attn_out = self.attn(wins)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x_attn = window_reverse(attn_out, self.window_size, H, W, pad)
        x = shortcut.permute(0,2,3,1) + x_attn
        x = x + self.mlp(self.norm2(x))
        return x.permute(0,3,1,2).contiguous()

class MainDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch//2,2,2,0)
        self.conv1 = ConvBNAct(in_ch//2+skip_ch, out_ch,3,1,1)
        self.conv2 = ConvBNAct(out_ch, out_ch,3,1,1)
        self.skip_ch = skip_ch
    def forward(self, x, skip_feat=None):
        x = self.upsample(x)
        if skip_feat is not None and self.skip_ch>0:
            if x.shape[2:] != skip_feat.shape[2:]:
                skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class FeatureMerge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,1,1,0,bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))
    def forward(self, main_feat, aux_feat):
        if main_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=main_feat.shape[2:], mode='bilinear', align_corners=False)
        if main_feat.shape[1] != aux_feat.shape[1]:
            aux_feat = nn.Conv2d(aux_feat.shape[1], main_feat.shape[1],1,1,0).to(main_feat.device)(aux_feat)
        return self.conv(torch.cat([main_feat, aux_feat], dim=1))

class Res16_DualDecoder_SegModel(nn.Module):
    def __init__(self, num_classes=6, swin_config=None):
        super().__init__()
        self.num_classes = num_classes
        self.swin_config = swin_config or SwinConfig()
        self.encoder = ResNet16Encoder()
        self.attention_skips = nn.ModuleList([SelfAttentionSkip(64), SelfAttentionSkip(128), SelfAttentionSkip(256), SelfAttentionSkip(512)])
        self.decoder_layers = [
            {"main":{"in":512,"skip":256,"out":256}, "aux":{"in":512,"out":256,"heads":2}, "merge":{"in":512,"out":256}},
            {"main":{"in":256,"skip":128,"out":128}, "aux":{"in":256,"out":128,"heads":4}, "merge":{"in":256,"out":128}},
            {"main":{"in":128,"skip":64,"out":64}, "aux":{"in":128,"out":64,"heads":8}, "merge":{"in":128,"out":64}},
            {"main":{"in":64,"skip":0,"out":64}, "aux":{"in":64,"out":64,"heads":16}, "merge":{"in":128,"out":64}},
        ]
        self.main_decoder = nn.ModuleList([MainDecoderBlock(d["main"]["in"],d["main"]["skip"],d["main"]["out"]) for d in self.decoder_layers])
        self.aux_decoder = nn.ModuleList([SwinDecoderBlock(d["aux"]["in"],d["aux"]["out"],d["aux"]["heads"],self.swin_config.window_size) for d in self.decoder_layers])
        self.merge_modules = nn.ModuleList([FeatureMerge(d["merge"]["in"],d["merge"]["out"]) for d in self.decoder_layers])
        self.final_upsample = nn.ConvTranspose2d(64,64,2,2,0)
        self.seg_head = nn.Sequential(ConvBNAct(64,32,3,1,1), nn.Dropout2d(0.15), nn.Conv2d(32, num_classes,1,1,0))
        self.apply(self._init_weights)
        print(f"Model initialization complete | Trainable parameters: {count_params(self):.2f}M")
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def forward(self, x):
        feat, skips = self.encoder(x)
        skips = [self.attention_skips[i](skips[i]) for i in range(4)]
        dec = feat
        idxs = [2,1,0,None]
        for i in range(4):
            sf = skips[idxs[i]] if idxs[i] is not None else None
            m = self.main_decoder[i](dec, sf)
            a = self.aux_decoder[i](dec, sf)
            dec = self.merge_modules[i](m,a)
        out = self.final_upsample(dec)
        return self.seg_head(out)

# ======================== 5. Main Test Function ========================
def main():
    PROJECT_ROOT = r'/root/autodl-tmp/ST-Unet'
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    test_output_dir = os.path.join(OUTPUT_DIR, DATASET_NAME)
    vis_dir = os.path.join(test_output_dir, 'visualization')
    pred_map_dir = os.path.join(test_output_dir, 'pred_maps')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_map_dir, exist_ok=True)
    print(f"📌 Test result save path: {test_output_dir}")

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    train_mean = np.array(train_config['dataset_stats']['mean'])
    train_std = np.array(train_config['dataset_stats']['std'])
    print(f"📊 Normalization parameters - mean: {train_mean.round(4)}, std: {train_std.round(4)}")

    def transform(img):
        img = img.astype(np.float32)
        img = (img - train_mean) / train_std
        return img

    test_dataset = SegTestDataset(root_path=ROOT_PATH, img_size=IMG_SIZE, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ===================== Load Model =====================
    print("\n🔧 Loading Res16_DualDecoder_SegModel...")
    model = Res16_DualDecoder_SegModel(num_classes=6).to(DEVICE)

    # Fixed weight loading error
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE, weights_only=False)

    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ Weights loaded successfully")
    except:
        model.load_state_dict(new_state_dict, strict=False)
        print("⚠️ Non-strict loading completed")

    model.eval()

    print("\n🔍 Verifying model output...")
    with torch.no_grad():
        test_out = model(torch.randn(1,3,256,256).to(DEVICE))
    print(f"✅ Model output shape: {test_out.shape}")

    metrics = SegMetrics(NUM_CLASSES)
    vis_count = 0

    print("\n🚀 Starting test...")
    with torch.no_grad():
        for imgs, masks, file_names in tqdm(test_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

            if vis_count < VIS_NUM:
                for i in range(len(file_names)):
                    if vis_count >= VIS_NUM: break
                    vis_path = os.path.join(vis_dir, f"{file_names[i]}.png")
                    vis_result(imgs[i], masks[i], preds[i], file_names[i], vis_path, train_mean, train_std)
                    pred = preds[i].cpu().numpy()
                    color = np.zeros((256,256,3), dtype=np.uint8)
                    for c in range(6):
                        color[pred==c] = CLASS_COLORS[c]
                    cv2.imwrite(os.path.join(pred_map_dir, f"{file_names[i]}_pred.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                    vis_count +=1

    results = metrics.compute()
    print("\n" + "="*60)
    print("📊 Test Results")
    print("="*60)
    print(f"OA: {results['OA']:.4f}")
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"mPrecision: {results['mPrecision']:.4f}")
    print(f"mRecall: {results['mRecall']:.4f}")
    print(f"mF1: {results['mF1']:.4f}")

    with open(os.path.join(test_output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()