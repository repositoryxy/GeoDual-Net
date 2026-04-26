import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------- Critical Fix: Add safe global objects --------------------------
import torch.serialization
torch.serialization.add_safe_globals([argparse.Namespace])

# -------------------------- Import Models (Added boundary-free branch ViT_seg) --------------------------
from models.SegViT_RS import SegViTRS, CFG as SegViTRS_CFG
from models.HRViT_RS import HRViTRS, CFG as HRViTRS_CFG
from models.DeepLabVp import DeeplabV3Plus
from models.SwinUnet import SwinUNet
from models.TransUnet import TransUNet
from models.UNet import UNet
from models.UperNet import UperNet
from modeling2 import CONFIGS as CONFIGS_ViT_seg

# -------------------------- Core Addition: Class to RGB color mapping --------------------------
LABEL_TO_RGB = {
    0: (0, 255, 255),    # Low Shrub -> Cyan
    1: (255, 255, 255),  # Impervious Surfaces -> White
    2: (0, 0, 255),      # Building -> Pure Blue
    3: (255, 0, 0),      # Bare Soil -> Pure Red
    4: (0, 255, 0),      # Vegetation -> Pure Green
    5: (255, 255, 0)     # Car -> Pure Yellow
}

def label_to_rgb(label_array):
    height, width = label_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for cls_idx, rgb in LABEL_TO_RGB.items():
        cls_mask = (label_array == cls_idx)
        rgb_image[cls_mask] = rgb
    return rgb_image

# -------------------------- Dataset Loading --------------------------
class RemoteSensingDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, list_path, img_size=256, mean=None, std=None):
        self.root_path = root_path
        self.img_size = img_size
        self.mean = mean
        self.std = std
        with open(list_path, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
        self.file_list = [fname.replace('.npz', '') for fname in self.file_list]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        npz_path = os.path.join(self.root_path, f"{file_name}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"File not found: {npz_path}")

        data = np.load(npz_path)
        img = data['image']
        label = data['label']

        # Save a copy of the original image for visualization (H,W,3)
        ori_img = img.copy()
        if ori_img.ndim == 3 and ori_img.shape[0] == 3:
            ori_img = np.transpose(ori_img, (1, 2, 0))
        if ori_img.dtype != np.uint8:
            ori_img = ori_img.astype(np.uint8)

        # Network input image processing
        if img.shape[0] == 3 and len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        img = self.to_tensor(img)
        mean = torch.tensor(self.mean, dtype=img.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=img.dtype).view(3, 1, 1)
        img = (img - mean) / std
        label = torch.from_numpy(label).long()

        return img, label, file_name, ori_img

# -------------------------- Metrics Calculation --------------------------
def compute_metrics(preds, labels, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(preds, labels):
        pred = pred.flatten()
        label = label.flatten()
        for p, l in zip(pred, label):
            if l < num_classes:
                confusion_matrix[l, p] += 1

    iou = []
    precision = []
    recall = []
    f1 = []

    for cls in range(num_classes):
        tp = confusion_matrix[cls, cls]
        fp = confusion_matrix[:, cls].sum() - tp
        fn = confusion_matrix[cls, :].sum() - tp

        iou_cls = tp / (tp + fp + fn + 1e-8)
        prec_cls = tp / (tp + fp + 1e-8)
        rec_cls = tp / (tp + fn + 1e-8)
        f1_cls = 2 * prec_cls * rec_cls / (prec_cls + rec_cls + 1e-8)

        iou.append(iou_cls)
        precision.append(prec_cls)
        recall.append(rec_cls)
        f1.append(f1_cls)

    total_correct = np.diag(confusion_matrix).sum()
    total_pixels = confusion_matrix.sum()
    OA = total_correct / total_pixels

    return {
        'confusion_matrix': confusion_matrix.tolist(),
        'IoU': [round(x, 4) for x in iou],
        'Precision': [round(x, 4) for x in precision],
        'Recall': [round(x, 4) for x in recall],
        'F1': [round(x, 4) for x in f1],
        'mIoU': round(np.mean(iou), 4),
        'mPrec': round(np.mean(precision), 4),
        'mRec': round(np.mean(recall), 4),
        'mF1': round(np.mean(f1), 4),
        'OA': round(OA, 4)
    }

# -------------------------- Main Testing Function --------------------------
def main():
    parser = argparse.ArgumentParser(description='Remote Sensing Image Segmentation Testing (Three-image visualization: Original + Ground Truth + Prediction)')
    parser.add_argument('--model_name', type=str, default='SegViTRS',
                        choices=[ 'SegViTRS',
                                 'HRViTRS','UNet', 'DeeplabV3Plus',
                                 'UperNet','SwinUNet','TransUNet'])
    parser.add_argument('--model_path', type=str,
                        default='')
    parser.add_argument('--root_path', type=str,
                        default='/root/autodl-tmp/Geo-SegViT/datasets/Potsdam/npz_data_RGB_improved')
    parser.add_argument('--list_path', type=str,
                        default='/root/autodl-tmp/Geo-SegViT/datasets/Potsdam/lists_txt_RGB_improved/test.txt')
    parser.add_argument('--data_stats_path', type=str,
                        default='/root/autodl-tmp/Geo-SegViT/datasets/Potsdam/rgb_data_stats_improved.npz')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--vit_patches_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_vis', action='store_true', default=True)
    parser.add_argument('--vis_save_dir', type=str,
                        default='/root/autodl-tmp/Geo-SegViT/TestResults/Geo-SegViT')
    parser.add_argument('--swin_embed_dim', type=int, default=64)
    parser.add_argument('--swin_window_size', type=int, default=4)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load mean and standard deviation
    stats = np.load(args.data_stats_path)
    train_mean = stats["mean"].tolist()
    train_std = stats["std"].tolist()
    train_mean = [x/255.0 for x in train_mean]
    train_std = [x/255.0 for x in train_std]

    # Dataset
    test_dataset = RemoteSensingDataset(
        root_path=args.root_path,
        list_path=args.list_path,
        img_size=args.img_size,
        mean=train_mean,
        std=train_std
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Number of test samples: {len(test_dataset)}")

    # Model initialization
    if   args.model_name == 'TransUNet':
        config_trans = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_trans.n_classes = args.num_classes
        config_trans.hidden_size = 768
        net = TransUNet(num_classes=args.num_classes, in_channels=3, img_size=args.img_size, config=config_trans).to(device)
    elif args.model_name == 'SegViTRS':
        cfg = SegViTRS_CFG.copy()
        cfg["in_channels"]=3; cfg["num_classes"]=args.num_classes
        net = SegViTRS(cfg, img_size=args.img_size).to(device)
    elif args.model_name == 'UNet':
        net = UNet(num_classes=args.num_classes, in_channels=3,features=[64, 128, 256, 256]).to(device)
    elif args.model_name == 'UperNet':
        net = UperNet(num_classes=args.num_classes, in_channels=3).to(device)
    elif args.model_name == 'HRViTRS':
        cfg = HRViTRS_CFG.copy()
        cfg["in_channels"]=3; cfg["num_classes"]=args.num_classes
        net = HRViTRS(cfg).to(device)
    elif args.model_name == 'DeeplabV3Plus':
        net = DeeplabV3Plus(num_classes=args.num_classes, in_channels=3).to(device)
    elif args.model_name == 'SwinUNet':
        net = SwinUNet(num_classes=args.num_classes, in_channels=args.in_channels,
                        img_size=args.img_size, embed_dim=96,
                        depths=tuple([2, 2, 6, 2]),
                        num_heads=tuple([3, 6, 12, 24])).to(device)
    else:
        raise ValueError("Unsupported model")

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}

    # ====== Core Fix: TransUNet weight name mapper ======
    if args.model_name == 'TransUNet':
        mapped_weights = {}
        for k, v in model_weights.items():
            # 1. Map feature_projection (Old independent layer -> New Sequential layer 0)
            if k == 'feature_projection.weight':
                mapped_weights['feature_projection.0.weight'] = v
            elif k == 'feature_projection.bias':
                mapped_weights['feature_projection.0.bias'] = v
            # 2. Map proj_norm (Old independent BN layer -> New Sequential layer 1)
            elif k == 'proj_norm.weight':
                mapped_weights['feature_projection.1.weight'] = v
            elif k == 'proj_norm.bias':
                mapped_weights['feature_projection.1.bias'] = v
            elif k == 'proj_norm.running_mean':
                mapped_weights['feature_projection.1.running_mean'] = v
            elif k == 'proj_norm.running_var':
                mapped_weights['feature_projection.1.running_var'] = v
            elif k == 'proj_norm.num_batches_tracked':
                mapped_weights['feature_projection.1.num_batches_tracked'] = v
            # 3. Filter out redundant decoder biases no longer needed in the new code
            elif k in ['decoder.0.bias', 'decoder.3.bias', 'decoder.6.bias']:
                continue
            else:
                mapped_weights[k] = v
        model_weights = mapped_weights

    # Load with strict=False (since we manually discarded useless biases)
    net.load_state_dict(model_weights, strict=False)
    net.eval()
    print("Model loaded successfully")

    # Inference
    preds_list, labels_list, file_names, ori_imgs_list = [], [], [], []
    with torch.no_grad():
        for imgs, labels, fnames, ori_imgs in test_loader:
            imgs = imgs.to(device)
            outputs = net(imgs)
            preds = torch.argmax(outputs, dim=1)
            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            file_names.extend(fnames)
            ori_imgs_list.append(ori_imgs.numpy() if isinstance(ori_imgs, torch.Tensor) else ori_imgs)

    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    ori_imgs = np.concatenate(ori_imgs_list, axis=0)

    # Metrics
    metrics = compute_metrics(preds, labels, args.num_classes)
    print("\n===== Test Results =====")
    print(f"mIoU: {metrics['mIoU']:.4f} | mF1: {metrics['mF1']:.4f} | OA: {metrics['OA']:.4f}")

    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.vis_save_dir, args.model_name, "Pots_256")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"metrics_{timestamp}.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    # ====================== Visualization: Three images side-by-side ======================
    if args.save_vis:
        vis_dir = os.path.join(result_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        total = len(preds)
        print(f"\nStarting to save {total} visualization images (Original + GT + Prediction)")

        for i in range(total):
            ori = ori_imgs[i]
            pred = preds[i]
            gt = labels[i]
            fname = file_names[i]

            pred_rgb = label_to_rgb(pred)
            gt_rgb = label_to_rgb(gt)

            plt.figure(figsize=(15, 5))
            # Original Image
            plt.subplot(1, 3, 1)
            plt.imshow(ori)
            plt.title("Original Image", fontsize=11)
            plt.axis('off')
            # Ground Truth
            plt.subplot(1, 3, 2)
            plt.imshow(gt_rgb)
            plt.title("Ground Truth", fontsize=11)
            plt.axis('off')
            # Prediction
            plt.subplot(1, 3, 3)
            plt.imshow(pred_rgb)
            plt.title("Prediction", fontsize=11)
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{fname}.png"), bbox_inches='tight', dpi=150, pad_inches=0.0)
            plt.close()

            if (i+1) % 20 ==0:
                print(f"Saved {i+1}/{total}")

        print(f"✅ Visualization completed, saved to: {vis_dir}")

if __name__ == '__main__':
    main()