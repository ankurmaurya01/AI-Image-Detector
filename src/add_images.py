"""
Dataset Ingestion Helper: Add and split new images into the database.
Allows adding individual images or an entire folder of images with automatic
format validation, collision-free naming, and optional auto-splitting (train/valid/test).
"""

import argparse
import os
import random
import shutil
import sys
from typing import List, Tuple
from PIL import Image

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")


def is_valid_image(filepath: str) -> bool:
    """Verify that an image file is readable and non-corrupt."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_current_counts(database_dir: str = "database") -> dict:
    """Return image counts across all splits and classes."""
    counts = {
        "train": {"real": 0, "fake": 0},
        "valid": {"real": 0, "fake": 0},
        "test": {"real": 0, "fake": 0},
    }
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(database_dir, split)
        if not os.path.exists(split_dir):
            continue
        for entry in os.listdir(split_dir):
            entry_path = os.path.join(split_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            entry_lower = entry.lower()
            if entry_lower in ["real", "fake"]:
                valid_files = [
                    f for f in os.listdir(entry_path)
                    if f.lower().endswith(VALID_EXTENSIONS)
                ]
                counts[split][entry_lower] = len(valid_files)
    return counts


def resolve_class_dir(database_dir: str, split: str, class_label: str) -> str:
    """Find existing real/fake folder regardless of case, or create one."""
    split_dir = os.path.join(database_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for entry in os.listdir(split_dir):
        if entry.lower() == class_label.lower() and os.path.isdir(os.path.join(split_dir, entry)):
            return os.path.join(split_dir, entry)

    # If doesn't exist, create with uppercase
    target = os.path.join(split_dir, class_label.upper())
    os.makedirs(target, exist_ok=True)
    return target


def copy_image(src_path: str, dest_dir: str) -> str:
    """Copy an image file into destination, ensuring no filename overwrite."""
    base_name = os.path.basename(src_path)
    name, ext = os.path.splitext(base_name)
    dest_path = os.path.join(dest_dir, base_name)

    counter = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
        counter += 1

    shutil.copy2(src_path, dest_path)
    return dest_path


def ingest_images(
    source_path: str,
    class_label: str,
    target_split: str = None,
    auto_split: bool = False,
    database_dir: str = "database",
) -> Tuple[int, int]:
    """
    Ingest images from source into database.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path not found: {source_path}")

    # Gather image files
    files_to_process = []
    if os.path.isfile(source_path):
        if source_path.lower().endswith(VALID_EXTENSIONS):
            files_to_process.append(source_path)
    else:
        for root, _, files in os.walk(source_path):
            for f in files:
                if f.lower().endswith(VALID_EXTENSIONS):
                    files_to_process.append(os.path.join(root, f))

    if not files_to_process:
        print(f"[WARNING] No valid image files found at: {source_path}")
        return 0, 0

    print(f"[INFO] Found {len(files_to_process)} candidate images. Validating integrity...")

    valid_images = []
    corrupt_count = 0
    for p in files_to_process:
        if is_valid_image(p):
            valid_images.append(p)
        else:
            corrupt_count += 1

    if corrupt_count > 0:
        print(f"[WARNING] Skipped {corrupt_count} corrupt or unreadable image files.")

    print(f"[INFO] {len(valid_images)} images passed validation.")

    # Shuffle for fair random distribution if auto-splitting
    if auto_split:
        random.seed(42)
        random.shuffle(valid_images)
        n = len(valid_images)
        n_train = int(n * 0.70)
        n_valid = int(n * 0.15)

        train_files = valid_images[:n_train]
        valid_files = valid_images[n_train:n_train + n_valid]
        test_files = valid_images[n_train + n_valid:]

        splits_map = {
            "train": train_files,
            "valid": valid_files,
            "test": test_files,
        }
    else:
        split_name = target_split or "train"
        splits_map = {split_name: valid_images}

    total_added = 0
    for split_name, files in splits_map.items():
        dest_dir = resolve_class_dir(database_dir, split_name, class_label)
        for f in files:
            copy_image(f, dest_dir)
            total_added += 1
        print(f"  -> Added {len(files)} images to '{dest_dir}'")

    return total_added, corrupt_count


def parse_args():
    parser = argparse.ArgumentParser(description="Add new images to the database")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image file or a directory containing new images",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        choices=["real", "fake"],
        help="Label of the new images: 'real' or 'fake'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Target split (default: 'train'). Ignored if --auto-split is used.",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        help="Automatically distribute images across splits: 70%% train, 15%% valid, 15%% test",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="database",
        help="Path to database root directory",
    )
    return parser.parse_args()


def print_summary_table(counts: dict, title: str):
    print("\n" + "=" * 50)
    print(f"       {title}")
    print("=" * 50)
    print(f"{'Split':<10} {'Real':<10} {'Fake':<10} {'Total':<10}")
    print("-" * 50)
    grand_total = 0
    for split in ["train", "valid", "test"]:
        r = counts[split]["real"]
        f = counts[split]["fake"]
        tot = r + f
        grand_total += tot
        print(f"{split.capitalize():<10} {r:<10} {f:<10} {tot:<10}")
    print("-" * 50)
    print(f"{'TOTAL':<10} {'':<10} {'':<10} {grand_total:<10}")
    print("=" * 50 + "\n")


def main():
    args = parse_args()

    before_counts = get_current_counts(args.database)
    print_summary_table(before_counts, "CURRENT DATABASE COUNTS")

    added, corrupt = ingest_images(
        source_path=args.source,
        class_label=args.label.lower(),
        target_split=args.split,
        auto_split=args.auto_split,
        database_dir=args.database,
    )

    after_counts = get_current_counts(args.database)
    print_summary_table(after_counts, "UPDATED DATABASE COUNTS")

    print(f"[SUCCESS] Successfully ingested {added} new '{args.label}' image(s).")
    print("Next steps:")
    print("  1. Fine-tune or retrain:  python src/train.py --weights models/best_model.pth --epochs 10")
    print("  2. Re-evaluate on test:  python src/evaluate.py")


if __name__ == "__main__":
    main()
