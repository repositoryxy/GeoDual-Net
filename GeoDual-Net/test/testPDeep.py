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

# ======================== 1. Core Configuration (aligned with model/training script) ========================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset configuration
DATASET_NAME = 'Pots_256'
ROOT_PATH = r'G:\erya308\zhangjunming\Geo-SegViT\datasets\Potsdam\npz_data_RGB_improved'
MODEL_WEIGHT_PATH = r"G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\FinalResultsP\DeeplabV3Plus\Pots_256\DeeplabV3Plus_Pots_256_256_bs8_ep100_final_20251209_114335.pth"
CONFIG_PATH = r"G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\FinalResultsP\DeeplabV3Plus\Pots_256\DeeplabV3Plus_Pots_256_256_bs8_ep100_config_20251209_114335.json"

# Hyperparameters (consistent with model/training)
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_CLASSES = 6
IN_CHANNELS = 3
VIS_NUM = 10
DEBUG = True

# Output configuration
OUTPUT_DIR = r'G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\TestResults\DeeplabV3Plus_P'

# Class configuration
CLASS_NAMES = [
    'Low shrub', 'Impervious surfaces', 'Building',
    'Background', 'Vegetation', 'Vehicle'
]
VALID_CLASSES = [0, 1, 2, 3, 4, 5]
CLASS_COLORS = [
    (0, 255, 255), (255, 255, 255), (0, 0, 255),
    (255, 0, 0), (0, 255, 0), (255, 255, 0)
]


# ======================== 2. Data loading (fixed all known errors) ========================
class DeeplabV3PlusTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, img_size=256, transform=None):
        self.root_path = root_path
        self.img_size = img_size
        self.transform = transform

        self.file_paths = []
        for f in os.listdir(root_path):
            if f.endswith('.npz'):
                self.file_paths.append(os.path.join(root_path, f))

        print(f"📁 Loaded {len(self.file_paths)} test samples")

        # Debug: count label range across full dataset
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

        # Load data
        data = np.load(npz_path)
        img = data['image']  # (3, 256, 256) → NumPy array
        mask = data['label']  # (256, 256) → NumPy array

        # Verify dimensions
        assert img.shape == (3, self.img_size, self.img_size), f"Image dimension error: {img.shape}"
        assert mask.shape == (self.img_size, self.img_size), f"Label dimension error: {mask.shape}"

        # Normalization
        if self.transform is not None:
            img_trans = img.transpose(1, 2, 0)  # (3,H,W) → (H,W,3)
            img_trans = self.transform(img_trans)
            img = img_trans.transpose(2, 0, 1)  # (H,W,3) → (3,H,W)

        # Debug: output first sample info
        if DEBUG and idx == 0:
            unique_labels = np.unique(mask)
            print(f"\n🔍 Sample {file_name} label distribution: {unique_labels}")
            print(f"🔍 Input image pixel range: [{img.min():.2f}, {img.max():.2f}]")
            print(f"🔍 Label value range: [{mask.min()}, {mask.max()}]")

        # Convert to tensor
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask, file_name


# ======================== 3. Evaluation Metrics ========================
class DeeplabV3PlusMetrics:
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

        # Per-class metrics
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

        # Mean metrics
        valid_mask = np.array([cls in VALID_CLASSES for cls in range(self.num_classes)])
        results['mIoU'] = float(np.mean(np.array(iou)[valid_mask]))
        results['mPrecision'] = float(np.mean(np.array(precision)[valid_mask]))
        results['mRecall'] = float(np.mean(np.array(recall)[valid_mask]))
        results['mF1'] = float(np.mean(np.array(f1)[valid_mask]))

        # Overall accuracy OA
        total_tp = np.diag(cm).sum()
        total_samples = cm.sum()
        results['OA'] = float(total_tp / (total_samples + 1e-8))

        # Per-class detailed metrics
        results['per_class_iou'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(iou)}
        results['per_class_precision'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(precision)}
        results['per_class_recall'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(recall)}
        results['per_class_f1'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(f1)}

        # Debug: output confusion matrix
        if DEBUG:
            print("\n🔍 Confusion matrix:")
            print(cm)

        return results


# ======================== 4. Visualization functions (adapted for model output) ========================
def vis_deeplabv3plus_result(img, mask, pred, file_name, save_path, mean, std):
    # Denormalize (adapted for mean type)
    img_np = img.cpu().numpy().transpose(1, 2, 0)  # (3,H,W) → (H,W,3)
    img_np = img_np * std + mean

    # Adjust denormalization based on mean range
    if mean.max() > 1:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # Convert label to color image (compatible with Tensor/NumPy)
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

    # Plot and save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(mask_color)
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('DeeplabV3Plus Prediction', fontsize=12)
    axes[2].axis('off')

    plt.suptitle(file_name, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ======================== 5. Main function (core adapted for model initialization) ========================
def main():
    # Add project root directory
    PROJECT_ROOT = r'G:\erya308\zhangjunming\Geo-SegViT'
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        print(f"✅ Added project root: {PROJECT_ROOT}")

    # Create output directories
    test_output_dir = os.path.join(OUTPUT_DIR, DATASET_NAME)
    vis_dir = os.path.join(test_output_dir, 'visualization')
    pred_map_dir = os.path.join(test_output_dir, 'pred_maps')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_map_dir, exist_ok=True)
    print(f"📌 Test result save path: {test_output_dir}")

    # Load training config and normalization parameters
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    train_mean = np.array(train_config['dataset_stats']['mean'])
    train_std = np.array(train_config['dataset_stats']['std'])
    print(f"📊 Normalization parameters - mean: {train_mean.round(4)}, std: {train_std.round(4)}")

    # Data preprocessing (adapted for mean type)
    def transform(img):
        img = img.astype(np.float32)
        # If mean is in 0-255 range, no need to divide by 255
        if train_mean.max() > 1:
            img = (img - train_mean) / train_std
        else:
            img = img / 255.0
            img = (img - train_mean) / train_std
        return img

    # Load dataset
    test_dataset = DeeplabV3PlusTestDataset(
        root_path=ROOT_PATH,
        img_size=IMG_SIZE,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    # Load model (exactly matching your model definition)
    print("\n🔧 Loading DeeplabV3Plus model...")
    from model2.DeepLabVp import DeeplabV3Plus  # Import your model class
    # Pass only supported parameters: num_classes + in_channels
    model = DeeplabV3Plus(
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS
    ).to(DEVICE)

    # Load weights
    try:
        checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)

    # Handle multi-GPU weight prefix
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Verify weight loading
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ Weights loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"⚠️ Strict weight loading failed: {e}")
        print("🔄 Trying non-strict loading...")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    # Verify model output validity
    print("\n🔍 Verifying model weight validity...")
    test_input = torch.randn(1, 3, 256, 256).to(DEVICE)
    with torch.no_grad():
        test_output = model(test_input)
    output_mean = test_output.mean().item()
    output_std = test_output.std().item()
    print(f"Random input model output mean: {output_mean:.4f}, std: {output_std:.4f}")
    if output_std < 1e-3:
        print("❌ Model weight abnormal! Output has no variance")
    else:
        print("✅ Model weight verification passed")

    # Initialize metrics
    metrics = DeeplabV3PlusMetrics(NUM_CLASSES)
    metrics.reset()

    # Start testing
    print("\n🚀 Starting DeeplabV3Plus model test...")
    vis_count = 0
    with torch.no_grad():
        for batch_idx, (imgs, masks, file_names) in enumerate(tqdm(test_loader, desc='DeeplabV3Plus test progress')):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward inference (model returns single output)
            outputs = model(imgs)  # (B, 6, 256, 256)
            preds = torch.argmax(outputs, dim=1)  # (B, 256, 256)

            # Debug: first batch prediction distribution
            if DEBUG and batch_idx == 0:
                unique_preds = np.unique(preds.cpu().numpy())
                print(f"🔍 First batch predicted classes: {unique_preds}")
                output_probs = F.softmax(outputs[0], dim=0).cpu().numpy()
                print(f"🔍 First sample prediction probabilities: {[round(output_probs[i].mean(), 4) for i in range(6)]}")

            # Update metrics
            metrics.update(preds, masks)

            # Visualization
            if vis_count < VIS_NUM:
                for i in range(len(file_names)):
                    if vis_count >= VIS_NUM:
                        break
                    vis_save_path = os.path.join(vis_dir, f"{file_names[i]}.png")
                    vis_deeplabv3plus_result(
                        imgs[i], masks[i], preds[i],
                        file_names[i], vis_save_path,
                        train_mean, train_std
                    )
                    # Save prediction map
                    pred = preds[i].cpu().numpy()
                    pred_color = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    for cls in range(NUM_CLASSES):
                        pred_color[pred == cls] = CLASS_COLORS[cls]
                    pred_save_path = os.path.join(pred_map_dir, f"{file_names[i]}_pred.png")
                    cv2.imwrite(pred_save_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                    vis_count += 1

    # Compute and save metrics
    results = metrics.compute()
    print("\n" + "=" * 60)
    print("📊 DeeplabV3Plus Model Test Results Summary")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Overall Accuracy (OA): {results['OA']:.4f}")
    print(f"Mean IoU (mIoU): {results['mIoU']:.4f}")
    print(f"Mean Precision: {results['mPrecision']:.4f}")
    print(f"Mean Recall: {results['mRecall']:.4f}")
    print(f"Mean F1 Score: {results['mF1']:.4f}")

    # Print per-class metrics
    print("\n📋 Per-class detailed metrics:")
    for cls_name in CLASS_NAMES:
        print(f"\n{cls_name}:")
        print(f"  IoU: {results['per_class_iou'][cls_name]:.4f}")
        print(f"  Precision: {results['per_class_precision'][cls_name]:.4f}")
        print(f"  Recall: {results['per_class_recall'][cls_name]:.4f}")
        print(f"  F1: {results['per_class_f1'][cls_name]:.4f}")

    # Save metrics to JSON
    metrics_path = os.path.join(test_output_dir, 'deeplabv3plus_test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Generate test report
    report = f"""
# DeeplabV3Plus Model Test Report
## Test Configuration
- Dataset: {DATASET_NAME}
- Number of samples: {len(test_dataset)}
- Input size: {IMG_SIZE}×{IMG_SIZE}
- Batch Size: {BATCH_SIZE}
- Input channels: {IN_CHANNELS}
- Model weights: {os.path.basename(MODEL_WEIGHT_PATH)}
- Normalization mean: {train_mean.round(4).tolist()}
- Normalization std: {train_std.round(4).tolist()}

## Core Metrics
| Metric | Value |
|--------|-------|
| OA (Overall Accuracy) | {results['OA']:.4f} |
| mIoU (mean IoU) | {results['mIoU']:.4f} |
| Mean Precision | {results['mPrecision']:.4f} |
| Mean Recall | {results['mRecall']:.4f} |
| Mean F1 Score | {results['mF1']:.4f} |

## Per-class IoU
"""
    for i, cls_name in enumerate(CLASS_NAMES):
        report += f"- {cls_name}: {results['per_class_iou'][cls_name]:.4f}\n"

    report_path = os.path.join(test_output_dir, 'deeplabv3plus_test_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ DeeplabV3Plus test completed!")
    print(f"📁 Metrics file: {metrics_path}")
    print(f"📄 Report file: {report_path}")
    print(f"🎨 Visualization results: {vis_dir}")
    print(f"🗺️ Prediction maps: {pred_map_dir}")


if __name__ == "__main__":
    main()