import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
import os
# Get project root path (ST-UNet path)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)  # Add root directory to environment variables

# Solve PyTorch 2.6+ weight loading security mechanism
import torch.serialization

torch.serialization.add_safe_globals([argparse.Namespace])

# Set Chinese font for visualization (remove, as we are translating to English)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------- Core configuration (fully aligned with training) --------------------------
# Label mapping configuration
RGB_LABEL_MAPPING = {
    (0, 255, 255): 0,  # Low shrub (cyan)
    (255, 255, 255): 1,  # Impervious surfaces (white)
    (0, 0, 255): 2,  # Building (pure blue)
    (255, 0, 0): 3,  # Background (pure red)
    (0, 255, 0): 4,  # Vegetation (pure green)
    (255, 255, 0): 5  # Vehicle (pure yellow)
}

CLASS_NAMES = [
    'Low shrub',  # 0 - Low shrub
    'Impervious surfaces',  # 1 - Impervious surfaces
    'Building',  # 2 - Building
    'Background',  # 3 - Background
    'Vegetation',  # 4 - Vegetation
    'Vehicle'  # 5 - Vehicle
]

CLASS_COLORS = [
    (0, 255, 255),  # 0 - cyan - Low shrub
    (255, 255, 255),  # 1 - white - Impervious surfaces
    (0, 0, 255),  # 2 - blue - Building
    (255, 0, 0),  # 3 - red - Background
    (0, 255, 0),  # 4 - green - Vegetation
    (255, 255, 0)  # 5 - yellow - Vehicle
]

# All classes (full 6-class training)
ALL_CLASSES = [0, 1, 2, 3, 4, 5]

# -------------------------- Configuration parameters --------------------------
parser = argparse.ArgumentParser()
# Data configuration
parser.add_argument('--root_path', type=str,
                    default=r'G:\erya308\zhangjunming\Geo-SegViT\datasets\Vaihingen\npz_data_RGB_improved',
                    help='Test data root directory')
parser.add_argument('--list_dir', type=str,
                    default=r'G:\erya308\zhangjunming\Geo-SegViT\datasets\Vaihingen\lists_txt_RGB_improved',
                    help='Test data list directory')
parser.add_argument('--img_size', type=int, default=256, help='Input image size')
parser.add_argument('--batch_size', type=int, default=8, help='Test batch size')
parser.add_argument('--num_classes', type=int, default=6, help='Number of classes')

# Model configuration (fixed to TransUNet)
parser.add_argument('--model_weight_path', type=str, required=True,
                    help='TransUNet model weight path')
parser.add_argument('--config_path', type=str, required=True,
                    help='Training configuration file path (.json)')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                    help='ViT model name (consistent with training)')
parser.add_argument('--n_skip', type=int, default=3,
                    help='Number of skip connections in TransUNet (consistent with training)')
parser.add_argument('--vit_patches_size', type=int, default=16,
                    help='ViT patch size (consistent with training)')

# Output configuration
parser.add_argument('--output_dir', type=str,
                    default=r'G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\TestResults',
                    help='Test result output directory')
parser.add_argument('--vis_num', type=int, default=10,
                    help='Number of samples for visualization')
parser.add_argument('--save_pred_maps', action='store_true', default=True,
                    help='Whether to save predicted segmentation maps')

args = parser.parse_args()


# -------------------------- Data loading class --------------------------
class VaihingenDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, list_dir, split='test', img_size=256, transform=None):
        self.root_path = root_path
        self.split = split
        self.img_size = img_size
        self.transform = transform

        # Scan the directory to load all npz files (ensure completeness)
        self.file_list = []
        for f in os.listdir(root_path):
            if f.endswith('.npz'):
                self.file_list.append(os.path.splitext(f)[0])
        print(f"📁 Scanned {len(self.file_list)} npz files")

        # Filter valid files
        self.valid_files = []
        self.missing_files = []
        for file_name in self.file_list:
            npz_path = os.path.join(root_path, f"{file_name}.npz")
            if os.path.exists(npz_path):
                self.valid_files.append(file_name)
            else:
                self.missing_files.append(file_name)

        self.file_list = self.valid_files
        print(f"✅ Number of valid samples: {len(self.valid_files)}")
        if self.missing_files:
            print(f"⚠️ Number of missing files: {len(self.missing_files)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        npz_path = os.path.join(self.root_path, f"{file_name}.npz")

        # Load data
        data = np.load(npz_path)
        img = data['image']  # (3,256,256)
        mask = data['label']  # (256,256)

        # Adjust image dimensions to (H,W,3)
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        # Normalization (consistent with training)
        if self.transform is not None:
            img = self.transform(img)

        # Convert to Tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return img, mask, file_name


# -------------------------- Evaluation metrics computation class --------------------------
class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """Update confusion matrix (full class computation)"""
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # Filter valid labels
        mask = (target >= 0) & (target < self.num_classes)
        pred = pred[mask]
        target = target[mask]

        if len(pred) > 0 and len(target) > 0:
            self.confusion_matrix += confusion_matrix(
                target, pred, labels=range(self.num_classes)
            )

    def compute(self):
        """Compute all-class metrics"""
        cm = self.confusion_matrix.copy()

        # Compute IoU per class
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = intersection / (union + 1e-6)

        # Compute Precision/Recall/F1 per class
        precision = intersection / (cm.sum(axis=0) + 1e-6)
        recall = intersection / (cm.sum(axis=1) + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Overall accuracy (OA)
        oa = np.diag(cm).sum() / (cm.sum() + 1e-6)

        # mean IoU (mIoU)
        miou = np.mean(iou)

        # mean Precision/Recall/F1
        mprecision = np.mean(precision)
        mrecall = np.mean(recall)
        mf1 = np.mean(f1)

        return {
            'confusion_matrix': cm.tolist(),
            'per_class_iou': {CLASS_NAMES[i]: float(iou[i]) for i in range(self.num_classes)},
            'per_class_precision': {CLASS_NAMES[i]: float(precision[i]) for i in range(self.num_classes)},
            'per_class_recall': {CLASS_NAMES[i]: float(recall[i]) for i in range(self.num_classes)},
            'per_class_f1': {CLASS_NAMES[i]: float(f1[i]) for i in range(self.num_classes)},
            'mIoU': float(miou),
            'OA': float(oa),
            'mPrecision': float(mprecision),
            'mRecall': float(mrecall),
            'mF1': float(mf1)
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


# -------------------------- Visualization function --------------------------
def vis_segmentation(img, mask, pred, file_name, save_path, mean, std):
    """Visualize original image, ground truth, and prediction"""
    # Denormalize image
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert labels to color images
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()

    def mask_to_color(mask_data, colors):
        h, w = mask_data.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(len(colors)):
            color_img[mask_data == i] = colors[i]
        return color_img

    mask_color = mask_to_color(mask, CLASS_COLORS)
    pred_color = mask_to_color(pred, CLASS_COLORS)

    # Plot and save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.suptitle(file_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# -------------------------- Main test function --------------------------
def main():
    # 1. Create output directories
    output_dir = os.path.join(args.output_dir, 'TransUNet')
    vis_dir = os.path.join(output_dir, 'visualization')
    pred_dir = os.path.join(output_dir, 'pred_maps')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # 2. Load training configuration
    with open(args.config_path, 'r', encoding='utf-8') as f:
        train_config = json.load(f)

    # 3. Build normalization function (exactly the same as training)
    train_mean = np.array(train_config['dataset_stats']['mean'])
    train_std = np.array(train_config['dataset_stats']['std'])

    def transform(img):
        img = img.astype(np.float32)
        mean = train_mean.reshape(1, 1, 3)
        std = train_std.reshape(1, 1, 3)
        img = (img - mean) / std
        return img

    # 4. Load dataset
    dataset = VaihingenDataset(
        root_path=args.root_path,
        list_dir=args.list_dir,
        split='test',
        img_size=args.img_size,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"\n📊 Dataset statistics:")
    print(f"Total test samples: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # 5. Initialize TransUNet model (exactly the same as training)
    print("\n📌 Initializing TransUNet model...")
    from model2.TransUnet import TransUNet
    from modelingnew import CONFIGS as CONFIGS_ViT_seg

    # Load ViT configuration
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )

    # Build model
    model = TransUNet(
        config=config_vit,
        num_classes=args.num_classes,
        in_channels=3,
        img_size=args.img_size
    ).to(device)

    # Load model weights
    checkpoint = torch.load(args.model_weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Handle 'module.' prefix from multi-GPU training
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print(f"✅ Model weights loaded: {args.model_weight_path}")

    # 6. Initialize evaluation metrics
    metrics = SegmentationMetrics(args.num_classes)

    # 7. Start testing
    print("\n🚀 Starting test...")
    vis_count = 0
    with torch.no_grad():
        for imgs, masks, file_names in tqdm(dataloader, desc='Test progress'):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Forward inference
            outputs = model(imgs)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            # Update metrics
            metrics.update(preds, masks)

            # Save prediction maps and visualizations
            for i in range(len(file_names)):
                # Save color prediction map
                if args.save_pred_maps:
                    pred = preds[i].cpu().numpy()
                    pred_color = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
                    for c in range(args.num_classes):
                        pred_color[pred == c] = CLASS_COLORS[c]
                    save_path = os.path.join(pred_dir, f"{file_names[i]}_pred.png")
                    cv2.imwrite(save_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Generate visualization comparison
                if vis_count < args.vis_num:
                    vis_path = os.path.join(vis_dir, f"{file_names[i]}_vis.png")
                    vis_segmentation(imgs[i], masks[i], preds[i], file_names[i], vis_path, train_mean, train_std)
                    vis_count += 1

    # 8. Compute and save metrics
    results = metrics.compute()
    print("\n" + "=" * 70)
    print("📊 TransUNet Test Results Summary (Full 6-class evaluation)")
    print("=" * 70)
    print(f"Overall Accuracy (OA): {results['OA']:.4f}")
    print(f"mean IoU (mIoU): {results['mIoU']:.4f}")
    print(f"mean Precision (mPrecision): {results['mPrecision']:.4f}")
    print(f"mean Recall (mRecall): {results['mRecall']:.4f}")
    print(f"mean F1 Score (mF1): {results['mF1']:.4f}")

    print("\n📋 Per-class detailed metrics:")
    for cls_name in CLASS_NAMES:
        print(f"\n{cls_name}:")
        print(f"  IoU: {results['per_class_iou'][cls_name]:.4f}")
        print(f"  Precision: {results['per_class_precision'][cls_name]:.4f}")
        print(f"  Recall: {results['per_class_recall'][cls_name]:.4f}")
        print(f"  F1: {results['per_class_f1'][cls_name]:.4f}")

    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Metrics saved to: {metrics_path}")

    # Generate test report
    report = f"""# TransUNet Test Report
## Test Configuration
- Dataset: Vaihingen (test set)
- Number of samples: {len(dataset)}
- Input size: {args.img_size}×{args.img_size}
- Batch size: {args.batch_size}
- Model weights: {os.path.basename(args.model_weight_path)}

## Core Metrics (Full 6-class evaluation)
| Metric | Value |
|--------|-------|
| OA (Overall Accuracy) | {results['OA']:.4f} |
| mIoU (mean IoU) | {results['mIoU']:.4f} |
| mPrecision (mean Precision) | {results['mPrecision']:.4f} |
| mRecall (mean Recall) | {results['mRecall']:.4f} |
| mF1 (mean F1 Score) | {results['mF1']:.4f} |

## Per-class IoU
"""
    for i, cls_name in enumerate(CLASS_NAMES):
        report += f"- {cls_name}: {results['per_class_iou'][cls_name]:.4f}\n"

    report_path = os.path.join(output_dir, 'test_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Test report saved to: {report_path}")
    print(f"\n🎉 Test completed! All results saved to: {output_dir}")


if __name__ == "__main__":
    main()