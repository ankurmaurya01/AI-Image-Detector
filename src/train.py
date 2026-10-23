"""
Training script for the AI-Generated Image Detector.
Performs training with validation after each epoch, model checkpointing,
learning rate scheduling, and metric logging.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

# Add parent directory to path to allow running as `python src/train.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_data_loader, get_train_val_loaders
from src.model import build_model
from src.utils import get_device, load_config, plot_training_curves, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AI-Generated Image Detector")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Override model backbone (e.g. resnet50, efficientnet_b0)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to an existing checkpoint (.pth) to resume or fine-tune from",
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    Execute one training epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch [{epoch}/{total_epochs}] Train",
        leave=False,
        unit="batch",
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        current_loss = running_loss / total
        current_acc = 100.0 * correct / total
        progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    Evaluate model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch [{epoch}/{total_epochs}] Valid",
        leave=False,
        unit="batch",
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            current_loss = running_loss / total
            current_acc = 100.0 * correct / total
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train(config: Dict[str, Any], weights_path: str = None) -> None:
    # 1. Setup seed and device
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    device = get_device(config["training"].get("device", "auto"))
    print(f"[INFO] Using computation device: {device}")

    # 2. DataLoaders
    print("[INFO] Preparing training and validation data loaders...")
    try:
        train_loader, valid_loader = get_train_val_loaders(config)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Dataset error:\n{e}\n")
        print("Please add images to your database folders before initiating training.")
        sys.exit(1)

    print(
        f"[INFO] Dataset loaded: {len(train_loader.dataset)} training samples, "
        f"{len(valid_loader.dataset)} validation samples."
    )

    # 3. Model construction
    print(
        f"[INFO] Building model: {config['model']['backbone']} "
        f"(pretrained={config['model']['pretrained']})..."
    )
    model = build_model(config).to(device)

    # Load existing checkpoint if specified
    if weights_path and os.path.exists(weights_path):
        print(f"[INFO] Resuming / fine-tuning from existing weights: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # 4. Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()

    lr = float(config["training"].get("learning_rate", 0.0001))
    weight_decay = float(config["training"].get("weight_decay", 0.0001))
    opt_name = config["training"].get("optimizer", "adamw").lower()

    if opt_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 5. Learning rate scheduler
    sched_name = config["training"].get("scheduler", "cosine").lower()
    num_epochs = int(config["training"].get("num_epochs", 20))

    if sched_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif sched_name == "steplr":
        step_size = config["training"].get("step_size", 7)
        gamma = config["training"].get("gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None

    # Checkpoint directory
    checkpoint_path = config["paths"].get("best_model_path", "models/best_model.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    start_time = time.time()
    print(f"\n[INFO] Starting training for {num_epochs} epochs...\n")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )

        # Validation phase
        valid_loss, valid_acc = validate(
            model, valid_loader, criterion, device, epoch, num_epochs
        )

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
            f"Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save checkpoint if best validation accuracy achieved
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_acc": valid_acc,
                "valid_loss": valid_loss,
                "config": config,
                "class_to_idx": config["dataset"].get("class_to_idx", {"real": 0, "fake": 1}),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  >>> [BEST] Saved new best checkpoint to {checkpoint_path} (Acc: {valid_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n[INFO] Training complete in {total_time / 60:.2f} minutes.")
    print(f"[INFO] Best Validation Accuracy: {best_val_acc:.2f}%")

    # Save training history JSON
    history_path = config["paths"].get("history_save_path", "models/training_history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Training history saved to: {history_path}")

    # Plot training curves
    curves_path = os.path.join(
        config["paths"].get("model_save_dir", "models"), "training_curves.png"
    )
    plot_training_curves(history, curves_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Apply optional command line overrides
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.backbone is not None:
        config["model"]["backbone"] = args.backbone

    train(config, weights_path=args.weights)


if __name__ == "__main__":
    main()
