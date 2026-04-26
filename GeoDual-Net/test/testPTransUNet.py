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

# Dataset configuration
DATASET_NAME = 'Vai_256'
ROOT_PATH = r'/root/autodl-tmp/Geo-SegViT/datasets/Vaihingen/npz_data_RGB_improved'
DATA_STATS_PATH = r"/root/autodl-tmp/Geo-SegViT/datasets/Vaihingen/rgb_data_stats_improved.npz"

# Model configuration
MODEL_NAME = 'TransUNet'
MODEL_WEIGHT_PATH = r"/root/autodl-tmp/Geo-SegViT/ComparedModels_U/TransUNet_V/best.pth"
# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 2
NUM_CLASSES = 6
IN_CHANNELS = 3
VIS_NUM = 100
DEBUG = True

# TransUNet parameters
VIT_NAME = 'R50-ViT-B_16'
VIT_PATCH_SIZE = 16
N_SKIP = 3
HIDDEN_SIZE = 768

# Output configuration
OUTPUT_DIR = r'/root/autodl-tmp/Geo-SegViT/TestResults/TransUNet_V'

# Class configuration
CLASS_NAMES = [
    'Low shrub', 'Impervious surfaces', 'Building',
    'Background', 'Vegetation', 'Vehicle'
]
VALID_CLASSES = [0, 1, 2, 3, 4, 5]
CLASS_COLORS = [
    (0, 255, 255),
    (255, 255, 255),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0)
]

# ======================== 2. Data Loading ========================
class TransUNetTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, img_size=256, transform=None):
        self.root_path = root_path
        self.img_size = img_size
        self.transform = transform

        self.file_paths = []
        for f in os.listdir(root_path):
            if f.endswith('.npz'):
                self.file_paths.append(os.path.join(root_path, f))

        print(f"📁 Loaded {len(self.file_paths)} test samples")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz_path = self.file_paths[idx]
        file_name = os.path.basename(npz_path).replace('.npz', '')

        data = np.load(npz_path)
        img = data['image']
        mask = data['label']

        assert img.shape == (3, self.img_size, self.img_size)
        assert mask.shape == (self.img_size, self.img_size)

        if self.transform is not None:
            img_trans = img.transpose(1, 2, 0).astype(np.float32)
            img_trans = self.transform(img_trans)
            img = img_trans.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        if DEBUG and idx == 0:
            unique_labels = np.unique(mask.numpy())
            print(f"🔍 Sample {file_name} label distribution: {unique_labels}")

        return img, mask, file_name

# ======================== 3. Evaluation Metrics ========================
class TransUNetMetrics:
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

            if len(pred_valid) > 0:
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

# ======================== 4. Visualization Function ========================
def vis_transunet_result(img, mask, pred, file_name, save_path, mean, std):
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

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
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask_color)
    axes[1].set_title('Label')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('TransUNet Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ======================== Main Function ========================
def main():
    PROJECT_ROOT = r'/root/autodl-tmp/Geo-SegViT'
    sys.path.insert(0, PROJECT_ROOT)

    test_output_dir = os.path.join(OUTPUT_DIR, DATASET_NAME)
    vis_dir = os.path.join(test_output_dir, 'visualization')
    pred_map_dir = os.path.join(test_output_dir, 'pred_maps')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_map_dir, exist_ok=True)
    print(f"📌 Test result save path: {test_output_dir}")

    # Load mean and std
    stats = np.load(DATA_STATS_PATH)
    train_mean = stats['mean']
    train_std = stats['std']
    print(f"📊 Mean: {train_mean.round(4)}")
    print(f"📊 Std: {train_std.round(4)}")

    # ✅ Exactly the same as DeepLab / your training code
    def transform(img):
        return (img - train_mean) / train_std

    test_dataset = TransUNetTestDataset(
        root_path=ROOT_PATH,
        img_size=IMG_SIZE,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Load model
    print("\n🔧 Loading TransUNet model...")
    from model2.TransUnet import TransUNet
    from modelingnew import CONFIGS as CONFIGS_ViT_seg

    config_vit = CONFIGS_ViT_seg[VIT_NAME]
    config_vit.n_classes = NUM_CLASSES
    config_vit.hidden_size = HIDDEN_SIZE
    config_vit.n_skip = N_SKIP
    config_vit.patches.grid = (
        int(IMG_SIZE / VIT_PATCH_SIZE),
        int(IMG_SIZE / VIT_PATCH_SIZE)
    )

    model = TransUNet(
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        img_size=IMG_SIZE,
        config=config_vit
    ).to(DEVICE)

    # Load weights
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(new_state_dict, strict=True)
    except:
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    print("✅ Model loaded successfully")

    metrics = TransUNetMetrics(NUM_CLASSES)
    vis_count = 0

    with torch.no_grad():
        for imgs, masks, file_names in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

            if vis_count < VIS_NUM:
                for i in range(len(file_names)):
                    if vis_count >= VIS_NUM:
                        break
                    vis_transunet_result(
                        imgs[i], masks[i], preds[i],
                        file_names[i],
                        os.path.join(vis_dir, f"{file_names[i]}.png"),
                        train_mean, train_std
                    )
                    vis_count += 1

    results = metrics.compute()
    print("\n" + "=" * 60)
    print("📊 TransUNet Test Results")
    print("=" * 60)
    print(f"OA: {results['OA']:.4f}")
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"mF1: {results['mF1']:.4f}")

    with open(os.path.join(test_output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()