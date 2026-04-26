import argparse
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import json
from datetime import datetime

# Set GPU (single card comparison, ensure fairness)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ======================== Import new models (core addition) ========================
# Import custom Res16_DualDecoder_SegModel
from model2.GeoDual_Net import Res16_DualDecoder_SegModel, SwinConfig

# Import existing models
from modelingnew import CONFIGS as CONFIGS_ViT_seg
from model2.UNet import UNet
from model2.UperNet import UperNet
from model2.TransUnet import TransUNet
from model2.SwinUnet import SwinUNet
from model2.DeepLabVp import DeeplabV3Plus

# Import training function
from tr_new2 import trainer_synapse


# ======================== Logging configuration ========================
def setup_logger(args, dataset_name):
    """Configure logging, output to both console and file"""
    log_dir = os.path.join(args.final_result_dir, args.model_name, dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.model_name}_train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file, encoding='utf-8')  # File output
        ]
    )
    return logging.getLogger(__name__)


# ======================== Argument parsing ========================
parser = argparse.ArgumentParser()
# Basic configuration (shared by all models)
parser.add_argument('--root_path', type=str,
                    default=r'/root/autodl-tmp/ST-Unet/datasets/Potsdam/npz_data_RGB_improved',
                    help='Data root directory')
parser.add_argument('--dataset', type=str, default='Remote sensing image semantic segmentation comparison experiment', help='Experiment name')
parser.add_argument('--list_dir', type=str,
                    default=r'/root/autodl-tmp/ST-Unet/datasets/Potsdam/lists_txt_RGB_improved',
                    help='Data list directory')
parser.add_argument('--num_classes', type=int, default=6, help='Number of output classes')
parser.add_argument('--max_iterations', type=int, default=30000, help='Maximum number of iterations')
parser.add_argument('--max_epochs', type=int, default=150, help='Maximum number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU (takes effect when specified via command line)')
parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs (fixed to single card)')
parser.add_argument('--deterministic', type=int, default=1, help='Whether to enable deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='Learning rate (unified for all models)')
parser.add_argument('--img_size', type=int, default=256, help='Input image size (unified for all models)')
parser.add_argument('--seed', type=int, default=1234, help='Random seed (ensures reproducibility)')
parser.add_argument('--n_skip', type=int, default=3, help='Number of skip connections (only for ViT_seg)')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='ViT model name (only for ViT_seg)')
parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT patch size (only for ViT_seg)')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None, help='Attention type (used by specific models)')

# Model selection parameter (added Res16_DualDecoder)
parser.add_argument('--model_name', type=str,
                    default='Res16_DualDecoder',  # Default to training the new model
                    choices=['UNet', 'UperNet', 'TransUNet',
                             'SwinUNet', 'DeeplabV3Plus', 'Res16_DualDecoder'],  # Added option
                    help='Select the model to train')

# Data statistics file path
parser.add_argument('--data_stats_path', type=str,
                    default=r"/root/autodl-tmp/ST-Unet/datasets/Potsdam/rgb_data_stats_improved.npz",
                    help='Path to data mean/std file')

# Final result saving configuration
parser.add_argument('--final_result_dir', type=str,
                    default=r"/root/autodl-tmp/ST-Unet/ComparedModel2/FinalResults_U",
                    help='Root directory for saving final results')
parser.add_argument('--save_best_only', action='store_true', default=False,
                    help='Whether to save only the best model (otherwise save the final epoch model)')

# New model specific parameters (optional, for customizing Swin configuration)
parser.add_argument('--swin_embed_dim', type=int, default=64, help='Swin decoder embedding dimension (only for Res16_DualDecoder)')
parser.add_argument('--swin_window_size', type=int, default=4, help='Swin window size (only for Res16_DualDecoder)')

args = parser.parse_args()


# ======================== Result saving function ========================
def save_final_results(args, net, snapshot_path, dataset_name, train_mean, train_std, final_metrics=None):
    """Save final training results: model weights, configuration file, performance metrics"""
    # 1. Create final results directory
    final_dir = os.path.join(args.final_result_dir, args.model_name, dataset_name)
    os.makedirs(final_dir, exist_ok=True)

    # 2. Generate unique identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    core_info = f"{args.model_name}_{dataset_name}_{args.img_size}_bs{args.batch_size}_ep{args.max_epochs}"

    # 3. Save model weights
    try:
        if args.save_best_only:
            # Find best model
            best_ckpt_files = [f for f in os.listdir(snapshot_path) if f.startswith('best_')]
            if best_ckpt_files:
                best_ckpt = sorted(best_ckpt_files)[-1]
                best_ckpt_path = os.path.join(snapshot_path, best_ckpt)
                final_ckpt_path = os.path.join(final_dir, f"{core_info}_best_{timestamp}.pth")
                shutil.copy2(best_ckpt_path, final_ckpt_path)
                logging.info(f"✅ Saved best model to: {final_ckpt_path}")
            else:
                logging.warning("Best model file not found, saving final epoch model")
                final_model_path = os.path.join(final_dir, f"{core_info}_final_{timestamp}.pth")
                torch.save({
                    'model': net.state_dict(),
                    'epoch': args.max_epochs,
                    'args': args,
                    'final_metrics': final_metrics,
                    'train_finish_time': timestamp
                }, final_model_path)
                logging.info(f"✅ Saved final model weights to: {final_model_path}")
        else:
            # Save final epoch model
            final_model_path = os.path.join(final_dir, f"{core_info}_final_{timestamp}.pth")
            torch.save({
                'model': net.state_dict(),
                'epoch': args.max_epochs,
                'args': args,
                'final_metrics': final_metrics,
                'train_finish_time': timestamp
            }, final_model_path)
            logging.info(f"✅ Saved final model weights to: {final_model_path}")
    except Exception as e:
        logging.error(f"Failed to save model weights: {str(e)}")
        raise

    # 4. Save training configuration
    try:
        config_path = os.path.join(final_dir, f"{core_info}_config_{timestamp}.json")
        serializable_args = {
            k: v for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, list, tuple))
        }
        serializable_args.update({
            'train_finish_time': timestamp,
            'dataset_stats': {'mean': train_mean, 'std': train_std},
            'dataset_name': dataset_name
        })
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_args, f, ensure_ascii=False, indent=4)
        logging.info(f"✅ Saved training configuration to: {config_path}")
    except Exception as e:
        logging.error(f"Failed to save configuration file: {str(e)}")
        raise

    # 5. Save performance metrics
    if final_metrics is not None:
        try:
            metrics_path = os.path.join(final_dir, f"{core_info}_metrics_{timestamp}.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(final_metrics, f, ensure_ascii=False, indent=4)
            logging.info(f"✅ Saved performance metrics to: {metrics_path}")
        except Exception as e:
            logging.error(f"Failed to save performance metrics: {str(e)}")
            raise

    return final_dir


# ======================== Main training logic ========================
if __name__ == "__main__":
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
    args.img_size = 256  # Force uniform image size
    args.batch_size = 8  # Force uniform batch size
    dataset_name = 'Pots_256'  # Can be switched to 'Pots_256'
    dataset_config = {
        'Vai_256': {
            'root_path': r'/root/autodl-tmp/ST-Unet/datasets/Vaihingen/npz_data_RGB_improved',
            'list_dir': r'/root/autodl-tmp/ST-Unet/datasets/Vaihingen/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3
        },
        'Pots_256': {
            'root_path': r'/root/autodl-tmp/ST-Unet/datasets/Potsdam/npz_data_RGB_improved',
            'list_dir': r'/root/autodl-tmp/ST-Unet/datasets/Potsdam/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3
        },
    }
    # Update dataset parameters
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.in_channels = dataset_config[dataset_name]['in_channels']
    args.is_pretrain = False

    # Initialize logs
    logger = setup_logger(args, dataset_name)
    logger.info(f"Start training {args.model_name} model, dataset: {dataset_name}")

    # Load data statistics
    if not os.path.exists(args.data_stats_path):
        raise FileNotFoundError(f"Data statistics file does not exist: {args.data_stats_path}")
    stats = np.load(args.data_stats_path)
    train_mean = stats["mean"].tolist()
    train_std = stats["std"].tolist()
    logger.info(f"✅ Data statistics loaded:")
    logger.info(f"   Mean (R/G/B): {[round(x, 4) for x in train_mean]}")
    logger.info(f"   Std (R/G/B): {[round(x, 4) for x in train_std]}")

    # Build snapshot save path
    model_folder = f"{args.model_name}_networks"
    core_tag = f"{args.model_name}_{dataset_name}_{args.img_size}"
    vit_tag = ""
    if args.model_name == 'ViT_seg':
        vit_tag = f"vit{args.vit_name}_skip{args.n_skip}_patch{args.vit_patches_size}"
    # New model specific tag (optional)
    new_model_tag = ""
    if args.model_name == 'Res16_DualDecoder':
        new_model_tag = f"swin_embed{args.swin_embed_dim}_win{args.swin_window_size}"

    # Common hyperparameter tags
    iter_tag = f"iter{args.max_iterations // 1000}k"
    epoch_tag = f"epo{args.max_epochs}"
    batch_tag = f"bs{args.batch_size}"
    lr_tag = f"lr{args.base_lr}"
    seed_tag = f"s{args.seed}"
    param_tag = "_".join([iter_tag, epoch_tag, batch_tag, lr_tag, seed_tag])

    # Concatenate path (filter empty strings)
    path_parts = [
        "/root/autodl-tmp/ST-Unet/ComResult_U",
        model_folder,
        core_tag,
        vit_tag,
        new_model_tag,
        param_tag
    ]
    path_parts = [p for p in path_parts if p.strip()]
    snapshot_path = os.path.normpath(os.path.join(*path_parts))

    logger.info('-------------------------------------------')
    logger.info(f"Current model: {args.model_name}")
    logger.info(f"Snapshot save path: {snapshot_path}")
    logger.info('-------------------------------------------')
    os.makedirs(snapshot_path, exist_ok=True)

    # ======================== Model initialization (core addition) ========================
    if args.model_name == 'UNet':
        net = UNet(in_channels=args.in_channels, num_classes=args.num_classes, features=[64, 128, 256, 256]).cuda()

    elif args.model_name == 'DeeplabV3Plus':
        net = DeeplabV3Plus(num_classes=args.num_classes, in_channels=args.in_channels).cuda()

    elif args.model_name == 'UperNet':
        net = UperNet(num_classes=args.num_classes, in_channels=args.in_channels).cuda()

    elif args.model_name == 'TransUNet':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        net = TransUNet(config=config_vit, num_classes=args.num_classes,
                        in_channels=args.in_channels, img_size=args.img_size).cuda()

    elif args.model_name == 'SwinUNet':
        net = SwinUNet(num_classes=args.num_classes, in_channels=args.in_channels,
                       img_size=args.img_size, embed_dim=96,
                       depths=tuple([2, 2, 6, 2]), num_heads=tuple([3, 6, 12, 24])).cuda()

    # -------------------- New model initialization (core addition) --------------------
    elif args.model_name == 'Res16_DualDecoder':
        # Initialize Swin configuration (customizable via command line)
        swin_config = SwinConfig()
        swin_config.embed_dim = args.swin_embed_dim
        swin_config.window_size = args.swin_window_size

        # Initialize new model
        net = Res16_DualDecoder_SegModel(
            num_classes=args.num_classes,
            swin_config=swin_config
        ).cuda()
        logger.info(f"✅ New model Res16_DualDecoder initialization complete, Swin configuration:")
        logger.info(f"   embed_dim: {swin_config.embed_dim}, window_size: {swin_config.window_size}")

    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # ======================== Resume training from checkpoint ========================
    start_epoch = 0
    if os.path.exists(snapshot_path):
        ckpt_files = [f for f in os.listdir(snapshot_path)
                      if f.startswith('RGBepoch_') and f.endswith('.pth')]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            latest_ckpt = ckpt_files[0]
            latest_ckpt_path = os.path.join(snapshot_path, latest_ckpt)
            logger.info(f"Found latest checkpoint for {args.model_name}: {latest_ckpt_path}")

            checkpoint = torch.load(latest_ckpt_path, map_location=torch.device('cuda'))
            net.load_state_dict(checkpoint['model'])
            current_epoch = int(latest_ckpt.split('_')[1].split('.')[0])
            start_epoch = current_epoch
            logger.info(f"Model state restored, will continue training from epoch {start_epoch}")
        else:
            logger.info(f"No historical checkpoint found for {args.model_name}, training from scratch")
    else:
        logger.info("Snapshot path does not exist, creating path and training from scratch")

    # ======================== Start training ========================
    logger.info(f"\n🚀 Starting {args.model_name} model training!")
    logger.info(f"   Dataset: {dataset_name} | Image size: {args.img_size}×{args.img_size}")
    logger.info(f"   Batch Size: {args.batch_size} | Learning rate: {args.base_lr}")
    logger.info(f"   Max Epoch: {args.max_epochs} | Max Iterations: {args.max_iterations}")

    # Execute training
    final_metrics = trainer_synapse(
        args,
        net,
        snapshot_path,
        start_epoch=start_epoch,
        train_mean=train_mean,
        train_std=train_std
    )

    # ======================== Save final results ========================
    logger.info("\n" + "=" * 50)
    logger.info("Saving final training results...")
    logger.info("=" * 50)
    save_final_results(args, net, snapshot_path, dataset_name, train_mean, train_std, final_metrics)
    logger.info("\n🎉 All final results saved successfully!")