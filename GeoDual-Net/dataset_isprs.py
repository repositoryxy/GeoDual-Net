import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


# -------------------------- Core fix: data augmentation function (ensure channel dimension is not disrupted) --------------------------
def random_rot_flip(image, label):
    # 1. First convert (C, H, W) to (H, W, C) (channels last to avoid spatial operations affecting channels)
    image = image.transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
    # 2. Perform rotation and flip (now only affects H/W dimensions, does not affect channels)
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)  # 0=horizontal flip, 1=vertical flip (only affects H/W)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    # 3. Convert back to (C, H, W) format
    image = image.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
    return image, label


def random_rotate(image, label):
    # 1. First convert (C, H, W) to (H, W, C)
    image = image.transpose(1, 2, 0)
    # 2. Perform rotation (only affects H/W dimensions)
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    # 3. Convert back to (C, H, W) format
    image = image.transpose(2, 0, 1)
    return image, label


# -------------------------- The rest of the code remains unchanged --------------------------
class RandomGenerator(object):
    def __init__(self, output_size=(256, 256), mean=None, std=None):
        self.output_size = output_size
        assert mean is not None and std is not None, "Must provide mean and std"
        assert len(mean) == 3 and len(std) == 3, "Mean and std must be 3-channel"
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label, case_name = sample['image'], sample['label'], sample['case_name']
        # print(f"[{case_name}] Initial dimensions: image={image.shape}, label={label.shape}")

        # Step 1: Force uniform dimensions to (C, H, W)
        if image.ndim == 3:
            if image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)  # (H,W,C)→(C,H,W)
                # print(f"[{case_name}] Dimension conversion: (H,W,C)→{image.shape}")
            elif image.shape[1] == 3:
                image = image.transpose(1, 0, 2)  # (H,C,W)→(C,H,W)
                # print(f"[{case_name}] Dimension conversion: (H,C,W)→{image.shape}")
            elif image.shape[0] == 3:
                # print(f"[{case_name}] Dimensions normal: {image.shape}")
                a=1
            else:
                raise ValueError(f"[{case_name}] 3D image without 3 channels: {image.shape}")
        else:
            raise ValueError(f"[{case_name}] Non-3D image: {image.ndim}D, shape {image.shape}")

        assert image.shape[0] == 3, f"[{case_name}] Channel count after unification != 3: {image.shape}"
        # print(f"[{case_name}] Unified dimensions: {image.shape}")

        # Step 2: Data augmentation (fixed, does not change dimension order)
        orig_image_shape = image.shape
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
            # print(f"[{case_name}] After rotation and flip: image={image.shape}, label={label.shape}")
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
            # print(f"[{case_name}] After random rotation: image={image.shape}, label={label.shape}")

        # Assertion will pass (dimensions unchanged after augmentation)
        assert image.shape == orig_image_shape, f"[{case_name}] Dimensions changed after augmentation: {orig_image_shape}→{image.shape}"

        # Step 3: Force resizing (channel dimension is not scaled)
        C, H, W = image.shape
        target_H, target_W = self.output_size
        img_zoom_ratio = (1.0, target_H / H, target_W / W)
        # print(f"[{case_name}] Before resize: {image.shape}, zoom ratio={img_zoom_ratio}")

        image = zoom(image, img_zoom_ratio, order=3)
        label = zoom(label, (target_H / H, target_W / W), order=0)

        assert image.shape[0] == 3, f"[{case_name}] Channel count != 3 after resize: {image.shape}"
        assert image.shape[1:] == (target_H, target_W), f"[{case_name}] Size after resize abnormal: {image.shape[1:]}≠{self.output_size}"
        # print(f"[{case_name}] After resize: image={image.shape}, label={label.shape}")

        # Step 4: Normalization
        for c in range(3):
            image[c, :, :] = (image[c, :, :] - self.mean[c]) / self.std[c]

        # Final Tensor dimensions
        image_tensor = torch.from_numpy(image.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.int64))
        # print(f"[{case_name}] Final Tensor dimensions: image={image_tensor.shape}, label={label_tensor.shape}\n")

        return {
            'image': image_tensor,
            'label': label_tensor,
            'case_name': case_name
        }


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = [line.strip() for line in open(os.path.join(list_dir, f"{self.split}.txt")).readlines()]
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        file_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, file_name)

        try:
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # print(f"[{file_name}] Loading raw data: image={image.shape}, label={label.shape}")
        except Exception as e:
            raise RuntimeError(f"[{file_name}] Failed to load: {str(e)}")

        # Label cleaning
        num_classes = 6
        label = np.where(label == 255, 0, label)
        label = np.clip(label, 0, num_classes - 1)

        sample = {'image': image, 'label': label, 'case_name': file_name}

        if self.transform:
            sample = self.transform(sample)

        return sample