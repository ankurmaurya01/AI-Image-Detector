"""
Inference script for the AI-Generated Image Detector.
Takes a single image path (or directory) and outputs 'Real' or 'AI-generated'
with confidence scores.
"""

import argparse
import os
import sys
from typing import Any, Dict, Tuple

from PIL import Image, UnidentifiedImageError
import torch

# Add parent directory to path to allow running as `python src/predict.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_transforms, safe_image_loader
from src.model import build_model
from src.utils import get_device, load_config


class DetectorPredictor:
    """
    Inference helper that wraps model loading, preprocessing, and prediction logic.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        weights_path: str,
        device_str: str = "auto",
    ):
        """
        Initialize the predictor with model weights and preprocessing transforms.

        Args:
            config: Loaded YAML configuration dictionary.
            weights_path: Path to trained checkpoint .pth file.
            device_str: Device setting ('auto', 'cuda', 'cpu').
        """
        self.config = config
        self.device = get_device(device_str)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at: '{weights_path}'.\n"
                "Please train the model first or provide a valid weights file via --weights."
            )

        # Load checkpoint with memory optimization
        checkpoint = torch.load(weights_path, map_location=self.device)
        saved_config = checkpoint.get("config", config)
        if isinstance(saved_config, dict) and "model" in saved_config:
            # Avoid downloading ImageNet weights since we load checkpoint weights directly
            saved_config = dict(saved_config)
            saved_config["model"] = dict(saved_config["model"])
            saved_config["model"]["pretrained"] = False
        self.model = build_model(saved_config)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        # Class mapping
        self.class_to_idx = checkpoint.get(
            "class_to_idx",
            config["dataset"].get("class_to_idx", {"real": 0, "fake": 1}),
        ) if isinstance(checkpoint, dict) else config["dataset"].get("class_to_idx", {"real": 0, "fake": 1})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Free checkpoint memory immediately
        del checkpoint
        import gc
        gc.collect()

        self.model.to(self.device)
        self.model.eval()

        self.transform = get_transforms(config, split="test")

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict whether a single image is Real or AI-generated.

        Args:
            image_path: Path to the input image file.

        Returns:
            Dictionary containing:
                - 'image_path': str
                - 'label': 'Real' or 'AI-generated'
                - 'confidence': float (percentage)
                - 'probabilities': dict of {class_name: float}
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: '{image_path}'")

        try:
            pil_img = safe_image_loader(image_path)
        except UnidentifiedImageError:
            raise ValueError(f"Unable to read or decode image at '{image_path}'. Ensure it is a valid image file.")
        except Exception as e:
            raise RuntimeError(f"Error reading image '{image_path}': {e}")

        # Preprocess and add batch dimension (1, C, H, W)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(torch.argmax(outputs, dim=1).item())
        raw_class = self.idx_to_class.get(pred_idx, "unknown")

        # Map internal class names to human-readable output
        display_label = "AI-generated" if raw_class.lower() == "fake" else "Real"
        confidence = float(probs[pred_idx] * 100.0)

        # Probabilities breakdown
        prob_breakdown = {}
        for idx, prob in enumerate(probs):
            name = self.idx_to_class.get(idx, f"Class {idx}")
            clean_name = "AI-generated" if name.lower() == "fake" else "Real"
            prob_breakdown[clean_name] = round(float(prob * 100.0), 2)

        return {
            "image_path": image_path,
            "label": display_label,
            "confidence": round(confidence, 2),
            "probabilities": prob_breakdown,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict if an image is Real or AI-generated")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to an image file (e.g. .jpg, .png, .webp)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained model weights checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'auto', 'cuda', 'cpu'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    weights_path = args.weights or config["paths"].get(
        "best_model_path", "models/best_model.pth"
    )

    try:
        predictor = DetectorPredictor(
            config=config,
            weights_path=weights_path,
            device_str=args.device,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] Model loading error:\n{e}")
        sys.exit(1)

    try:
        result = predictor.predict(args.image)
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        sys.exit(1)

    # Format result output
    print("\n" + "=" * 50)
    print("           DETECTION RESULT")
    print("=" * 50)
    print(f" Image       : {result['image_path']}")
    print(f" Prediction  : {result['label']}")
    print(f" Confidence  : {result['confidence']:.2f}%")
    print("-" * 50)
    print(" Probabilities Breakdown:")
    for cls_name, prob in result["probabilities"].items():
        print(f"   - {cls_name:<14}: {prob:.2f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
