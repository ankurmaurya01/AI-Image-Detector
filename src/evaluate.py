"""
Evaluation script for the AI-Generated Image Detector.
Evaluates a trained model checkpoint on the test set and calculates
accuracy, precision, recall, F1-score, and generates a confusion matrix.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path to allow running as `python src/evaluate.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_data_loader
from src.model import build_model
from src.utils import get_device, load_config, plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AI-Generated Image Detector")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained model weights checkpoint (.pth)",
    )
    return parser.parse_args()


def evaluate_model(
    config: Dict[str, Any],
    weights_path: str,
) -> Dict[str, Any]:
    """
    Run evaluation on the test set.

    Args:
        config: Loaded YAML configuration.
        weights_path: Path to checkpoint .pth file.

    Returns:
        Dictionary of computed evaluation metrics.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at '{weights_path}'.\n"
            "Please train the model first using 'python src/train.py' or specify a valid checkpoint."
        )

    device = get_device(config["training"].get("device", "auto"))
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading checkpoint from: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)

    # Use configuration from checkpoint if available, else current config
    saved_config = checkpoint.get("config", config)
    model = build_model(saved_config)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Load test set
    try:
        test_loader = get_data_loader("test", config, shuffle=False)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Test dataset error:\n{e}\n")
        print("Please add images to 'database/test/real' and 'database/test/fake' before evaluating.")
        sys.exit(1)

    print(f"[INFO] Evaluating on test set ({len(test_loader.dataset)} images)...")

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Resolve class names
    class_to_idx = config["dataset"].get("class_to_idx", {"real": 0, "fake": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)

    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("                EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"Macro Precision  : {precision_macro * 100:.2f}%")
    print(f"Macro Recall     : {recall_macro * 100:.2f}%")
    print(f"Macro F1-Score   : {f1_macro * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60 + "\n")

    # Save confusion matrix plot
    cm_path = config["paths"].get("confusion_matrix_path", "models/confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=class_names, output_path=cm_path)

    # Save metrics JSON
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "per_class": {
            class_names[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
    }

    metrics_save_path = os.path.join(
        config["paths"].get("model_save_dir", "models"), "evaluation_metrics.json"
    )
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Detailed evaluation metrics saved to: {metrics_save_path}")

    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    weights_path = args.weights or config["paths"].get(
        "best_model_path", "models/best_model.pth"
    )
    evaluate_model(config, weights_path)


if __name__ == "__main__":
    main()
