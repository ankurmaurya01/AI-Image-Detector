"""
Dataset loading, preprocessing, and augmentation pipeline.
Supports loading from database/train, database/valid, and database/test
with real and fake subfolders.
"""

import os
from typing import Any, Dict, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Supported image file extensions
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


def safe_image_loader(path: str) -> Image.Image:
    """
    Safely load an image file and convert it to 3-channel RGB.
    Handles RGBA, palette-based, grayscale, and varying image formats.

    Args:
        path: Path to the image file.

    Returns:
        PIL Image in RGB mode.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_transforms(config: Dict[str, Any], split: str = "train") -> transforms.Compose:
    """
    Construct torchvision transform pipelines based on split and config.

    Args:
        config: Loaded YAML configuration dictionary.
        split: One of 'train', 'valid', or 'test'.

    Returns:
        Composed torchvision transformations.
    """
    image_size = config["dataset"].get("image_size", 224)
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        aug_cfg = config.get("augmentation", {})
        flip_prob = aug_cfg.get("random_horizontal_flip_prob", 0.5)
        rot_deg = aug_cfg.get("random_rotation_degrees", 15)
        color_cfg = aug_cfg.get("color_jitter", {})

        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.RandomRotation(degrees=rot_deg),
            transforms.ColorJitter(
                brightness=color_cfg.get("brightness", 0.1),
                contrast=color_cfg.get("contrast", 0.1),
                saturation=color_cfg.get("saturation", 0.1),
                hue=color_cfg.get("hue", 0.05),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Validation, testing, and inference: deterministic resize and normalization
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class ConfiguredImageFolder(datasets.ImageFolder):
    """
    Subclass of torchvision ImageFolder that enforces deterministic
    class-to-index mapping from the configuration.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.custom_class_to_idx = class_to_idx
        super().__init__(root, transform=transform, loader=safe_image_loader)

    def find_classes(self, directory: str) -> Tuple[list, Dict[str, int]]:
        if self.custom_class_to_idx is not None:
            classes = sorted(list(self.custom_class_to_idx.keys()))
            return classes, self.custom_class_to_idx
        return super().find_classes(directory)


def count_valid_images(directory: str) -> int:
    """
    Count the number of valid image files inside a directory tree.
    """
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                count += 1
    return count


def get_data_loader(
    split: str,
    config: Dict[str, Any],
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Create a DataLoader for a specific split ('train', 'valid', 'test').

    Args:
        split: 'train', 'valid', or 'test'.
        config: Loaded YAML configuration dictionary.
        shuffle: Optional boolean to override default shuffle behavior.

    Returns:
        PyTorch DataLoader instance.

    Raises:
        FileNotFoundError: If the split directory does not exist or has 0 valid images.
    """
    dir_key = f"{split}_dir"
    split_dir = config["paths"].get(dir_key)
    if not split_dir:
        raise ValueError(f"Path key '{dir_key}' is missing in configuration 'paths' section.")

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory '{split_dir}' does not exist.")

    total_images = count_valid_images(split_dir)
    if total_images == 0:
        raise FileNotFoundError(
            f"No image files found in '{split_dir}'.\n"
            f"Please populate '{split_dir}/real' and '{split_dir}/fake' with "
            f"images (formats: {', '.join(VALID_IMAGE_EXTENSIONS)}) before running."
        )

    transform = get_transforms(config, split=split)
    class_to_idx = config["dataset"].get("class_to_idx", {"real": 0, "fake": 1})

    dataset = ConfiguredImageFolder(
        root=split_dir,
        transform=transform,
        class_to_idx=class_to_idx,
    )

    batch_size = config["dataset"].get("batch_size", 32)
    num_workers = config["dataset"].get("num_workers", 2)
    is_train = split == "train"
    should_shuffle = is_train if shuffle is None else shuffle

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


class TransformedDataset(torch.utils.data.Dataset):
    """Wraps a Subset/Dataset to apply split-specific transforms to raw PIL images."""

    def __init__(self, subset: torch.utils.data.Subset, transform: Optional[transforms.Compose] = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        img, target = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.subset)


def get_train_val_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """
    Get training and validation DataLoaders.
    If 'database/valid' has images, uses database/train and database/valid.
    If 'database/valid' is missing or empty, automatically splits 'database/train'
    into train and validation subsets using the configurable 'val_split_ratio' (default: 0.15).
    """
    valid_dir = config["paths"].get("valid_dir", "database/valid")
    has_valid_images = count_valid_images(valid_dir) > 0

    if has_valid_images:
        train_loader = get_data_loader("train", config, shuffle=True)
        valid_loader = get_data_loader("valid", config, shuffle=False)
        return train_loader, valid_loader

    # Fallback: Automatic train/val split from database/train
    train_dir = config["paths"].get("train_dir", "database/train")
    if not os.path.exists(train_dir) or count_valid_images(train_dir) == 0:
        raise FileNotFoundError(
            f"No training images found in '{train_dir}'. "
            f"Please populate '{train_dir}/real' and '{train_dir}/fake'."
        )

    val_split_ratio = float(config["dataset"].get("val_split_ratio", 0.15))
    class_to_idx = config["dataset"].get("class_to_idx", {"real": 0, "fake": 1})
    seed = config["training"].get("seed", 42)

    # Base dataset without transforms (returns raw PIL images)
    base_dataset = ConfiguredImageFolder(
        root=train_dir,
        transform=None,
        class_to_idx=class_to_idx,
    )

    total_len = len(base_dataset)
    val_len = max(1, int(total_len * val_split_ratio))
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        base_dataset, [train_len, val_len], generator=generator
    )

    train_transform = get_transforms(config, split="train")
    val_transform = get_transforms(config, split="valid")

    train_ds = TransformedDataset(train_subset, train_transform)
    val_ds = TransformedDataset(val_subset, val_transform)

    batch_size = config["dataset"].get("batch_size", 32)
    num_workers = config["dataset"].get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[INFO] 'database/valid' is empty. Automatically split '{train_dir}' "
        f"({int((1 - val_split_ratio) * 100)}% train [{train_len} images], "
        f"{int(val_split_ratio * 100)}% validation [{val_len} images])."
    )

    return train_loader, valid_loader


def get_all_data_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to build train, valid, and test loaders.

    Args:
        config: Loaded YAML configuration dictionary.

    Returns:
        Tuple of (train_loader, valid_loader, test_loader).
    """
    train_loader, valid_loader = get_train_val_loaders(config)
    test_loader = get_data_loader("test", config)
    return train_loader, valid_loader, test_loader
