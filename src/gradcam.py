"""
Explainable AI (XAI) using Gradient-weighted Class Activation Mapping (Grad-CAM).
Implements the interpretability method described in the CIFAKE research paper
(Bird & Lotfi, 2024), revealing which image regions and visual artifacts
contribute to the Real vs. AI-Generated prediction.
"""

import os
import sys
from typing import Optional, Tuple

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import safe_image_loader, get_transforms
from src.utils import load_config, get_device
from src.model import build_model


class GradCAM:
    """
    Grad-CAM implementation for convolutional neural networks.
    Computes class activation heatmaps highlighting regions that trigger predictions.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Automatically locate last conv layer if not specified
        if target_layer is None:
            target_layer = self._find_target_layer()

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional layer in ResNet or EfficientNet."""
        if hasattr(self.model, "backbone"):
            bb = self.model.backbone
            if hasattr(bb, "layer4"):  # ResNet
                return bb.layer4[-1]
            elif hasattr(bb, "features"):  # EfficientNet
                return bb.features[-1]

        # Fallback: search all modules for last Conv2d
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: Tensor of shape (1, C, H, W).
            target_class: Integer class index. If None, highest probability class is used.

        Returns:
            2D numpy array with values in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        score = output[0, target_class]
        score.backward()

        # Gradients pooling across spatial dimensions: (1, C, H, W) -> (C,)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of activation channels
        activations = self.activations[0]
        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU

        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    @staticmethod
    def overlay_heatmap(
        pil_img: Image.Image,
        heatmap: np.ndarray,
        colormap_name: str = "jet",
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Overlay a 2D heatmap on a PIL image.

        Args:
            pil_img: Original PIL Image.
            heatmap: 2D numpy array [0, 1].
            colormap_name: Matplotlib colormap name ('jet', 'plasma', 'viridis').
            alpha: Heatmap transparency factor (0.0 to 1.0).

        Returns:
            Blended PIL Image with heatmap overlay.
        """
        w, h = pil_img.size
        # Resize heatmap to match original image dimensions
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
        heatmap_resized = np.array(heatmap_img) / 255.0

        # Apply colormap (compatible with modern and older matplotlib)
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(colormap_name)
        except Exception:
            cmap = cm.get_cmap(colormap_name)
        colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # Discard alpha
        colored_heatmap = np.uint8(255 * colored_heatmap)

        # Blend with original image
        orig_np = np.array(pil_img.convert("RGB"))
        blended = np.uint8(alpha * colored_heatmap + (1 - alpha) * orig_np)

        return Image.fromarray(blended)


def explain_image(
    image_path: str,
    config_path: str = "config.yaml",
    weights_path: Optional[str] = None,
    output_path: Optional[str] = "models/gradcam_explanation.png",
) -> Tuple[Image.Image, dict]:
    """
    Run full Grad-CAM explanation on an image and save the visual explanation.

    Args:
        image_path: Path to target image.
        config_path: Path to configuration YAML.
        weights_path: Checkpoint path.
        output_path: Destination path for the explanation visualization.

    Returns:
        Tuple of (overlay_image, result_dict).
    """
    config = load_config(config_path)
    weights = weights_path or config["paths"].get("best_model_path", "models/best_model.pth")
    device = get_device("auto")

    checkpoint = torch.load(weights, map_location=device)
    saved_cfg = checkpoint.get("config", config)
    model = build_model(saved_cfg).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load and preprocess image
    pil_img = safe_image_loader(image_path)
    transform = get_transforms(config, split="test")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Forward pass for prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    class_to_idx = checkpoint.get("class_to_idx", config["dataset"].get("class_to_idx", {"real": 0, "fake": 1}))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    raw_class = idx_to_class.get(pred_idx, "unknown")
    display_label = "AI-generated" if raw_class.lower() == "fake" else "Real"
    confidence = float(probs[pred_idx] * 100.0)

    # Compute Grad-CAM
    cam = GradCAM(model)
    heatmap = cam.generate_heatmap(input_tensor, target_class=pred_idx)
    overlay_img = cam.overlay_heatmap(pil_img, heatmap, colormap_name="jet", alpha=0.55)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Create a side-by-side comparison: [Original, Raw Heatmap, Blended Overlay]
        w, h = pil_img.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(pil_img, (0, 0))
        combined.paste(overlay_img, (w, 0))
        combined.save(output_path)
        print(f"[INFO] Grad-CAM explanation saved to: {output_path}")

    result = {
        "label": display_label,
        "confidence": round(confidence, 2),
        "target_class_idx": pred_idx,
        "probabilities": {
            "Real": round(float(probs[class_to_idx.get("real", 0)] * 100.0), 2),
            "AI-generated": round(float(probs[class_to_idx.get("fake", 1)] * 100.0), 2),
        },
    }

    return overlay_img, result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Explain predictions using Grad-CAM")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="models/gradcam_explanation.png", help="Output path")
    args = parser.parse_args()

    overlay, res = explain_image(args.image, output_path=args.output)
    print(f"Prediction: {res['label']} ({res['confidence']}%)")
