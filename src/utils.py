"""
Utility functions for configuration loading, device management,
reproducibility seeding, and plotting metrics.
"""

import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and validate the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If essential sections are missing from the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate essential sections
    required_sections = ["paths", "dataset", "model", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file.")

    return config


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for deterministic execution.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_setting: str = "auto") -> torch.device:
    """
    Resolve the PyTorch device based on setting and hardware availability.

    Args:
        device_setting: One of 'auto', 'cuda', or 'cpu'.

    Returns:
        torch.device instance.
    """
    setting = device_setting.lower()
    if setting == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif setting == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("[WARNING] CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "models/confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    import matplotlib.pyplot as plt
    """
    Render and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array of shape (num_classes, num_classes).
        class_names: List of class labels (e.g. ['Real', 'AI-generated']).
        output_path: Destination path to save the image.
        title: Plot title.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    try:
        import seaborn as sns

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
    except ImportError:
        # Fallback to pure matplotlib if seaborn is not available
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {output_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: str = "models/training_curves.png",
) -> None:
    import matplotlib.pyplot as plt
    """
    Plot and save training/validation loss and accuracy curves across epochs.

    Args:
        history: Dictionary containing 'train_loss', 'valid_loss',
                 'train_acc', 'valid_acc'.
        output_path: Destination path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["valid_loss"], "r--s", label="Valid Loss")
    ax1.set_title("Loss across Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    ax2.plot(epochs, history["valid_acc"], "r--s", label="Valid Accuracy")
    ax2.set_title("Accuracy across Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Training curves saved to: {output_path}")
