import os
import numpy as np
import tifffile
from tqdm import tqdm

# -------------------------- 1. Configuration parameters (core modification: added test set ratio) --------------------------
RAW_DATA_ROOT = r"/root/autodl-tmp/ST-UNet/datasets/Potsdam"
IMAGE_DIR = os.path.join(RAW_DATA_ROOT, "2_Ortho_RGB")
LABEL_DIR = os.path.join(RAW_DATA_ROOT, "5_Labels_all")

IMAGE_SUFFIX = "_RGB"
OUTPUT_NPZ_DIR = os.path.join(RAW_DATA_ROOT, "npz_data_RGB_improved")
OUTPUT_LIST_DIR = os.path.join(RAW_DATA_ROOT, "lists_txt_RGB_improved")
PATCH_SIZE = 256
OVERLAP = 64
TRAIN_RATIO = 0.7  # Training set 70%
VAL_RATIO = 0.2    # Validation set 20%
TEST_RATIO = 0.1   # Test set 10% (new)
SAVE_STATS = True
MIN_VALID_PIXELS = 10
KNOWN_INVALID_VALUES = {-1, 127, 255}


# -------------------------- 2. Label mapping (unchanged) --------------------------
RGB_LABEL_MAPPING = {
    (0, 255, 255): 0,  # Low shrub (cyan)
    (255, 255, 255): 1,  # Impervious surface (white)
    (0, 0, 255): 2,  # Building (pure blue)
    (255, 0, 0): 3,  # Bare soil (pure red)
    (0, 255, 0): 4,  # Vegetation (pure green)
    (255, 255, 0): 5  # Vehicle (pure yellow)
}
INVALID_LABEL = 255
CLASS_NAMES = ["Low shrub-cyan", "Impervious surface-white", "Building-blue", "Bare soil-red", "Vegetation-green", "Vehicle-yellow"]


# -------------------------- 3. Utility functions (unchanged) --------------------------
def read_rgb_label(label_path):
    rgb_label = tifffile.imread(label_path)
    if rgb_label.ndim == 3:
        if rgb_label.shape[-1] != 3 and rgb_label.shape[0] == 3:
            rgb_label = np.transpose(rgb_label, (1, 2, 0))
    elif rgb_label.ndim == 2:
        rgb_label = np.stack([rgb_label] * 3, axis=-1)
    else:
        raise ValueError(f"Label dimension error: {rgb_label.ndim}D, path: {label_path}")

    h, w = rgb_label.shape[:2]
    class_label = np.full((h, w), INVALID_LABEL, dtype=np.uint8)
    for (r, g, b), idx in RGB_LABEL_MAPPING.items():
        mask = np.all(rgb_label == [r, g, b], axis=-1)
        class_label[mask] = idx

    invalid_ratio = np.sum(class_label == INVALID_LABEL) / (h * w) * 100
    if invalid_ratio > 10:
        print(f"Warning: {os.path.basename(label_path)} invalid pixel ratio {invalid_ratio:.2f}%")
    return class_label


def read_rgb_image(img_path):
    img = tifffile.imread(img_path)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"RGB image format error: {img.shape}, path: {img_path}")
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    if np.any(img < 0) or np.any(img > 255):
        print(f"Note: {os.path.basename(img_path)} pixel values outside 0-255 range, consider normalization")
    return img


def crop_to_patches(data, patch_size, overlap):
    patches = []
    c, h, w = data.shape
    stride = patch_size - overlap
    row_steps = max(1, (h - patch_size + stride) // stride)
    col_steps = max(1, (w - patch_size + stride) // stride)
    if row_steps * stride + patch_size < h:
        row_steps += 1
    if col_steps * stride + patch_size < w:
        col_steps += 1

    for i in range(row_steps):
        for j in range(col_steps):
            h_start = min(i * stride, h - patch_size)
            w_start = min(j * stride, w - patch_size)
            patch = data[:, h_start:h_start + patch_size, w_start:w_start + patch_size]
            patches.append(patch)
    return patches


def post_process_label(label_patch):
    for val in KNOWN_INVALID_VALUES:
        label_patch[label_patch == val] = INVALID_LABEL
    valid_mask = (label_patch >= 0) & (label_patch < len(RGB_LABEL_MAPPING))
    label_patch[~valid_mask] = INVALID_LABEL
    return label_patch


def calculate_class_distribution(npz_dir):
    class_counts = np.zeros(len(RGB_LABEL_MAPPING), dtype=np.uint64)
    npz_paths = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")]
    for npz_path in tqdm(npz_paths, desc="Analyzing RGB dataset class distribution"):
        data = np.load(npz_path)
        label = data["label"]
        for cls in range(len(RGB_LABEL_MAPPING)):
            class_counts[cls] += np.sum(label == cls)

    total = np.sum(class_counts)
    print("\n" + "=" * 60)
    print("RGB dataset class distribution statistics:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
        ratio = count / total * 100 if total > 0 else 0
        print(f"{name}({i}): {count} pixels, ratio {ratio:.2f}%")
    print("=" * 60 + "\n")

    if total > 0 and class_counts[5] / total < 0.01:
        print("Warning: Vehicle class has extremely low proportion. Suggestions:")
        print("1. Increase sampling weight for vehicle patches during training; 2. Use Focal Loss; 3. Apply vehicle sample augmentation\n")
    return class_counts


def calculate_data_stats(npz_dir):
    npz_paths = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")]
    if not npz_paths:
        raise ValueError("No RGB NPZ files, cannot compute statistics!")

    channel_means = np.zeros(3, dtype=np.float64)
    channel_vars = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for npz_path in tqdm(npz_paths, desc="Computing RGB data statistics (mean/std)"):
        data = np.load(npz_path)
        img = data["image"]
        pixels = img.size // 3
        total_pixels += pixels

        for c in range(3):
            channel_data = img[c].ravel()
            channel_means[c] += np.sum(channel_data)
            channel_vars[c] += np.sum(channel_data **2 - 2 * channel_data * channel_means[c] / total_pixels)

    channel_means /= total_pixels
    channel_vars = (channel_vars / total_pixels) + channel_means** 2
    channel_stds = np.sqrt(channel_vars)

    stats_path = os.path.join(os.path.dirname(npz_dir), "rgb_data_stats_improved.npz")
    np.savez(stats_path, mean=channel_means, std=channel_stds)
    print(f"\nRGB data statistics saved to: {stats_path}")
    print(f"R channel mean: {channel_means[0]:.4f}, std: {channel_stds[0]:.4f}")
    print(f"G channel mean: {channel_means[1]:.4f}, std: {channel_stds[1]:.4f}")
    print(f"B channel mean: {channel_means[2]:.4f}, std: {channel_stds[2]:.4f}\n")
    return channel_means, channel_stds


# -------------------------- 4. Main logic (core modification: split train/val/test) --------------------------
def main():
    os.makedirs(OUTPUT_NPZ_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LIST_DIR, exist_ok=True)
    npz_names = []
    invalid_samples = []
    class_patch_counts = np.zeros(len(RGB_LABEL_MAPPING), dtype=int)

    img_files = [f for f in os.listdir(IMAGE_DIR)
                 if f.endswith(".tif") and
                 not f.endswith(".tfw") and
                 IMAGE_SUFFIX in f]

    if not img_files:
        raise ValueError(f"No valid files in RGB image directory {IMAGE_DIR}! Please check: 1. Path correctness; 2. Filename contains {IMAGE_SUFFIX}")

    for img_name in tqdm(img_files, desc="Processing RGB image-label pairs"):
        if not img_name.endswith(f"{IMAGE_SUFFIX}.tif"):
            continue
        core_name = img_name.replace(f"{IMAGE_SUFFIX}.tif", "")
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, f"{core_name}_label.tif")

        if not os.path.exists(label_path):
            print(f"Warning: Label does not exist, skipping -> {os.path.basename(label_path)}")
            continue

        try:
            img_data = read_rgb_image(img_path)
            label_data = read_rgb_label(label_path)[np.newaxis, :, :]

            img_patches = crop_to_patches(img_data, PATCH_SIZE, OVERLAP)
            label_patches = crop_to_patches(label_data, PATCH_SIZE, OVERLAP)

            for idx, (img_patch, label_patch) in enumerate(zip(img_patches, label_patches)):
                clean_label = post_process_label(np.squeeze(label_patch))
                valid_pixels = np.sum(clean_label != INVALID_LABEL)
                has_vehicle = np.any(clean_label == 5)

                if valid_pixels < MIN_VALID_PIXELS and not has_vehicle:
                    continue
                if has_vehicle and valid_pixels < MIN_VALID_PIXELS:
                    print(f"Keeping low-valid-pixel patch containing vehicle: {core_name}_RGB_patch{idx}")

                for cls in range(len(RGB_LABEL_MAPPING)):
                    if np.any(clean_label == cls):
                        class_patch_counts[cls] += 1

                npz_name = f"{core_name}_RGB_patch{idx}.npz"
                npz_path = os.path.join(OUTPUT_NPZ_DIR, npz_name)
                np.savez_compressed(npz_path, image=img_patch, label=clean_label)
                npz_names.append(npz_name)

        except Exception as e:
            err_msg = f"Error processing {img_name}: {str(e)}"
            print(f"⚠️ {err_msg}, skipping")
            invalid_samples.append(img_name)
            continue

    if not npz_names:
        raise ValueError("No RGB NPZ files generated! Please check: 1. Image/label paths; 2. Label mapping correctness")

    print("\nRGB dataset patch counts per class:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name}: {class_patch_counts[i]} patches")

    # -------------------------- Core modification: 7:2:1 split logic --------------------------
    np.random.shuffle(npz_names)  # Randomly shuffle data
    total = len(npz_names)
    # Compute counts for each set (integer rounding by ratio)
    test_num = int(total * TEST_RATIO)
    val_num = int(total * VAL_RATIO)
    train_num = total - test_num - val_num  # Ensure sum equals total

    # Split sets
    test_list = npz_names[:test_num]                  # First 10% as test set
    val_list = npz_names[test_num:test_num+val_num]   # Next 20% as validation set
    train_list = npz_names[test_num+val_num:]         # Remaining 70% as training set

    # Save list files for three sets
    train_txt_path = os.path.join(OUTPUT_LIST_DIR, "train.txt")
    val_txt_path = os.path.join(OUTPUT_LIST_DIR, "val.txt")
    test_txt_path = os.path.join(OUTPUT_LIST_DIR, "test.txt")  # New test set file

    with open(train_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join([os.path.join(OUTPUT_NPZ_DIR, name) for name in train_list]))
    with open(val_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join([os.path.join(OUTPUT_NPZ_DIR, name) for name in val_list]))
    with open(test_txt_path, "w", encoding="utf-8") as f:  # Save test set
        f.write("\n".join([os.path.join(OUTPUT_NPZ_DIR, name) for name in test_list]))

    if SAVE_STATS:
        calculate_data_stats(OUTPUT_NPZ_DIR)
    calculate_class_distribution(OUTPUT_NPZ_DIR)

    # Output summary (including test set)
    print("\n" + "=" * 80)
    print("✅ RGB dataset preprocessing completed!")
    print(f"Total RGB NPZ files: {total}")
    print(f"Training set (70%): {len(train_list)} files -> path: {train_txt_path}")
    print(f"Validation set (20%): {len(val_list)} files -> path: {val_txt_path}")
    print(f"Test set (10%): {len(test_list)} files -> path: {test_txt_path}")  # New test set info
    if invalid_samples:
        print(f"Failed samples: {len(invalid_samples)} files -> examples: {invalid_samples[:3]}...")
    print("=" * 80)


if __name__ == "__main__":
    main()