import logging
import os
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision import transforms
from utils2 import DiceLoss
from dataset_isprs import Synapse_dataset, RandomGenerator


def cal_metrics(pred, label, num_classes):
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    tn = np.zeros(num_classes)

    with torch.no_grad():
        out = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        prediction = out.cpu().numpy()
        label = label.cpu().numpy()

        for cat in range(num_classes):
            tp[cat] += ((prediction == cat) & (label == cat)).sum()
            fp[cat] += ((prediction == cat) & (label != cat)).sum()
            fn[cat] += ((prediction != cat) & (label == cat)).sum()
            tn[cat] += ((prediction != cat) & (label != cat)).sum()

    precision = np.divide(tp, (tp + fp + 1e-8))
    recall = np.divide(tp, (tp + fn + 1e-8))
    iou = np.divide(tp, (tp + fp + fn + 1e-8))

    return tp, fp, fn, tn, precision, recall, iou


def init_result_csv(save_path, num_classes):
    headers = ['epoch', 'loss', 'mIoU', 'mPrecision', 'mRecall']
    for cls_idx in range(num_classes):
        headers.extend([f'class_{cls_idx}_IoU', f'class_{cls_idx}_Precision', f'class_{cls_idx}_Recall'])

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    logging.info(f"✅ Initialized training result CSV file: {save_path}")


def save_epoch_result_to_csv(save_path, epoch, loss, mean_iou, mean_prec, mean_recall, epoch_iou, epoch_prec, epoch_recall):
    data = [
        epoch, round(loss, 4), round(mean_iou, 4), round(mean_prec, 4), round(mean_recall, 4)
    ]
    for cls_idx in range(len(epoch_iou)):
        data.extend([round(epoch_iou[cls_idx], 4), round(epoch_prec[cls_idx], 4), round(epoch_recall[cls_idx], 4)])

    with open(save_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data)


# ====================== Training main function (optimized as requested) ======================
def trainer_synapse(args, model, snapshot_path, start_epoch=0, train_mean=None, train_std=None):
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"),
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(base_dir=args.root_path,
                               list_dir=args.list_dir,
                               split="train",
                               transform=transforms.Compose([
                                   RandomGenerator(
                                       output_size=[args.img_size, args.img_size],
                                       mean=train_mean,
                                       std=train_std
                                   )
                               ]))
    print(f"Training set size: {len(db_train)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn,
                             drop_last=True)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if start_epoch > 0:
        ckpt_files = [f for f in os.listdir(snapshot_path) if f.startswith('RGBepoch_') and f.endswith('.pth')]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            latest_ckpt_path = os.path.join(snapshot_path, ckpt_files[0])
            checkpoint = torch.load(latest_ckpt_path)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"✅ Optimizer state restored, starting from epoch {start_epoch}")

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    max_epoch = args.max_epochs
    len_trainloader = len(trainloader)
    max_iterations = max_epoch * len_trainloader
    completed_iter = start_epoch * len_trainloader
    iter_num = completed_iter

    result_csv_path = os.path.join(snapshot_path, 'train_results.csv')
    if not os.path.exists(result_csv_path) or start_epoch == 0:
        init_result_csv(result_csv_path, num_classes)

    # ====================== Save only the best model ======================
    best_mIoU = 0.0  # Track best mIoU
    best_epoch = 0   # Track best epoch

    for epoch_num in tqdm(range(start_epoch, max_epoch), ncols=70):
        model.train()
        epoch_loss = 0.0
        epoch_tp = np.zeros(num_classes)
        epoch_fp = np.zeros(num_classes)
        epoch_fn = np.zeros(num_classes)

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            epoch_loss += loss.item()

            tp, fp, fn, tn, precision, recall, iou = cal_metrics(outputs, label_batch, num_classes)
            epoch_tp += tp
            epoch_fp += fp
            epoch_fn += fn

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1

        # Compute metrics for current epoch
        epoch_iou = np.divide(epoch_tp, (epoch_tp + epoch_fp + epoch_fn + 1e-8))
        epoch_precision = np.divide(epoch_tp, (epoch_tp + epoch_fp + 1e-8))
        epoch_recall = np.divide(epoch_tp, (epoch_tp + epoch_fn + 1e-8))
        mean_iou = np.nanmean(epoch_iou)
        mean_precision = np.nanmean(epoch_precision)
        mean_recall = np.nanmean(epoch_recall)
        avg_epoch_loss = epoch_loss / len_trainloader
        current_epoch = epoch_num + 1

        # ====================== Log per epoch (retained) ======================
        logging.info(f"\n[Epoch {current_epoch}/{max_epoch}] "
                     f"Loss={avg_epoch_loss:.4f} | "
                     f"mIoU={mean_iou:.4f} | mPrec={mean_precision:.4f} | mRec={mean_recall:.4f}")

        for cls_idx in range(num_classes):
            logging.info(f"  Class {cls_idx}: IoU={epoch_iou[cls_idx]:.4f}, Prec={epoch_precision[cls_idx]:.4f}, Rec={epoch_recall[cls_idx]:.4f}")
            writer.add_scalar(f'metrics/class_{cls_idx}_IoU', epoch_iou[cls_idx], epoch_num)
            writer.add_scalar(f'metrics/class_{cls_idx}_Precision', epoch_precision[cls_idx], epoch_num)
            writer.add_scalar(f'metrics/class_{cls_idx}_Recall', epoch_recall[cls_idx], epoch_num)

        writer.add_scalar('metrics/mIoU', mean_iou, epoch_num)
        writer.add_scalar('metrics/mPrecision', mean_precision, epoch_num)
        writer.add_scalar('metrics/mRecall', mean_recall, epoch_num)
        writer.add_scalar('loss/epoch_loss', avg_epoch_loss, epoch_num)

        # Save metrics to CSV per epoch (retained)
        save_epoch_result_to_csv(
            result_csv_path, current_epoch, avg_epoch_loss,
            mean_iou, mean_precision, mean_recall,
            epoch_iou, epoch_precision, epoch_recall
        )

        # ====================== Save only the best model (core modification) ======================
        if mean_iou > best_mIoU:
            best_mIoU = mean_iou
            best_epoch = current_epoch
            save_path = os.path.join(snapshot_path, "best_model.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': current_epoch,
                'mIoU': best_mIoU
            }, save_path)
            logging.info(f"\n🎉 [Best model updated] Epoch {current_epoch} | mIoU = {best_mIoU:.4f}\n")

    # Output best results after training
    logging.info("\n" + "="*60)
    logging.info(f"🎉 Training completed! Total {max_epoch} epochs")
    logging.info(f"🏆 Best model at epoch {best_epoch}")
    logging.info(f"🎯 Best mIoU = {best_mIoU:.4f}")
    logging.info(f"📁 Best model saved to: {os.path.join(snapshot_path, 'best_model.pth')}")
    logging.info("="*60)

    writer.close()
    return "Training completed!"