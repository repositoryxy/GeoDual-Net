import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import other comparison models
from model2.HRViT_RS import HRViTRS
from models.SegViT_RS import SegViTRS

from tr_new2 import trainer_synapse


# Define custom argument parser
def parse_args():
    parser = argparse.ArgumentParser()

    # Basic configuration
    parser.add_argument('--root_path', type=str,
                        default='/root/autodl-tmp/ST-Unet/datasets/Vaihingen/npz_data_RGB_improved',
                        help='Data root directory')
    parser.add_argument('--dataset', type=str, default='Remote sensing image semantic segmentation comparison experiment', help='Experiment name')
    parser.add_argument('--list_dir', type=str,
                        default='/root/autodl-tmp/ST-Unet/datasets/Vaihingen/lists_txt_RGB_improved',
                        help='Data list directory')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of output classes')
    parser.add_argument('--max_iterations', type=int, default=30000, help='Maximum number of iterations')
    parser.add_argument('--max_epochs', type=int, default=150, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--deterministic', type=int, default=1, help='Deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--n_skip', type=int, default=3, help='Number of skip connections')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='ViT model name')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT patch size')
    parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None, help='Attention type')

    # Added: TransUNet specific pretrained weight path (extremely critical!)
    parser.add_argument('--pretrained_path', type=str,
                        default='/root/autodl-tmp/ST-Unet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz',
                        help='Absolute path to ViT pretrained weights')

    # Added: Regularization parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--drop_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.25, help='Drop path rate')

    # Model selection
    parser.add_argument('--model_name', type=str,
                        default='HRViTRS',
                        choices=['SegViTRS','HRViTRS'],
                        help='Select model to train')

    # Data statistics
    parser.add_argument('--data_stats_path', type=str,
                        default="/root/autodl-tmp/ST-Unet/datasets/Vaihingen/rgb_data_stats_improved.npz",
                        help='Data statistics file path')

    return parser.parse_args()


def get_model(args):
    """Get model based on model name"""
    if args.model_name == 'SegViTRS':
        cfg = {
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "patch_size": 4,
            "window_size": 8,
            "mlp_ratio": 4.0,
            "drop_rate": 0.1,
            "use_geo_pos_encoding": True,
            "use_land_prior": True,
            "decoder_embed_dim": 64
        }
        return SegViTRS(cfg=cfg, img_size=args.img_size).cuda()

    elif args.model_name == 'HRViTRS':
        CFG = {
            "in_channels": 3,  # Input channels (RGB remote sensing images)
            "num_classes": 6,  # 6-class output
            "embed_dims": [64, 128],  # Embedding dimensions for high-resolution streams
            "num_heads": [4, 8],  # Number of attention heads
            "window_size": 4,  # Window size adapted for remote sensing (4×4)
            "depths": [2, 2],  # Number of HRViT blocks for each resolution stream
            "drop_rate": 0.1,  # Dropout rate
            "use_spectral_attention": True,  # Remote sensing specific: spectral attention
            "use_geo_pos_encoding": True  # Remote sensing specific: geospatial position encoding
        }
        return HRViTRS(CFG).cuda()
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")


def get_trainer(args):
    """Get trainer"""
    if args.model_name == 'SegViTRS_Balanced':
        return trainer_synapse
    else:
        return trainer_synapse


def main():
    args = parse_args()

    # Deterministic training configuration
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Dataset configuration
    args.img_size = 256
    args.batch_size = 8
    dataset_name = 'Vai_256'

    dataset_config = {
        'Vai_256': {
            'root_path': '/root/autodl-tmp/ST-Unet/datasets/Vaihingen/npz_data_RGB_improved',
            'list_dir': '/root/autodl-tmp/ST-Unet/datasets/Vaihingen/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3
        },
        'Pots_256':{
            'root_path': '/root/autodl-tmp/ST-Unet/datasets/Potsdam/npz_data_RGB_improved',
            'list_dir': '/root/autodl-tmp/ST-Unet/datasets/Potsdam/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3
        }
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.in_channels = dataset_config[dataset_name]['in_channels']

    # Force enable pretraining
    args.is_pretrain = True

    # Load data statistics
    if not os.path.exists(args.data_stats_path):
        raise FileNotFoundError(f"Data statistics file does not exist: {args.data_stats_path}")

    stats = np.load(args.data_stats_path)
    train_mean = stats["mean"].tolist()
    train_std = stats["std"].tolist()

    print("=" * 50)
    print(f"✅ Data statistics loaded:")
    print(f"   Mean (R/G/B): {[round(x, 4) for x in train_mean]}")
    print(f"   Std (R/G/B): {[round(x, 4) for x in train_std]}")
    print("=" * 50)

    # Dynamically generate save path, fix hardcoded Pots issue
    dataset_prefix = dataset_name.split('_')[0]
    snapshot_path = os.path.join(
        "/root/autodl-tmp/ST-Unet/ComparedModels_U",
        f"{args.model_name}_{dataset_prefix}_networksD",
        f"{args.model_name}_{dataset_name}_{args.img_size}",
        f"iter{args.max_iterations // 1000}k_epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}_s{args.seed}"
    )

    os.makedirs(snapshot_path, exist_ok=True)

    print(f"📁 Model save path: {snapshot_path}")
    print("=" * 50)

    # Get model
    print(f"🔧 Initializing model: {args.model_name}")
    net = get_model(args)

    # Compute and display model parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"📊 Model parameter count:")
    print(f"   Total parameters: {total_params / 1e6:.2f} M")
    print(f"   Trainable parameters: {trainable_params / 1e6:.2f} M")
    print("=" * 50)

    # Checkpoint recovery
    start_epoch = 0
    latest_ckpt_path = None

    if os.path.exists(snapshot_path):
        ckpt_files = [f for f in os.listdir(snapshot_path)
                      if f.startswith('RGBepoch_') and f.endswith('.pth')]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            latest_ckpt = ckpt_files[0]
            latest_ckpt_path = os.path.join(snapshot_path, latest_ckpt)
            print(f"Found latest checkpoint for {args.model_name}: {latest_ckpt_path}")

            checkpoint = torch.load(latest_ckpt_path, map_location=torch.device('cuda'))
            net.load_state_dict(checkpoint['model'])
            current_epoch = int(latest_ckpt.split('_')[1].split('.')[0])
            start_epoch = current_epoch
            print(f"Model state restored, continuing from epoch {start_epoch}")
        else:
            print(f"No historical checkpoint found for {args.model_name}, training from scratch")
    else:
        print("Snapshot path does not exist, training from scratch and creating path")

    # Get trainer
    trainer = get_trainer(args)

    # Print training parameters
    print(f"🎯 Training parameters:")
    print(f"   Model: {args.model_name}")
    print(f"   Learning rate: {args.base_lr}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.max_epochs}")
    print(f"   Random seed: {args.seed}")
    print("=" * 50)

    # Start training
    print("🚀 Starting training...")
    print("=" * 50)

    trainer(
        args,
        net,
        snapshot_path,
        start_epoch=start_epoch,
        train_mean=train_mean,
        train_std=train_std
    )


if __name__ == "__main__":
    main()