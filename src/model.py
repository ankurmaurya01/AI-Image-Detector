"""
Deep learning model architectures for AI-Generated vs Real image detection.
Supports ResNet and EfficientNet backbones with configurable transfer learning
(pretrained weights) or training from scratch.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchvision import models


class AIImageDetector(nn.Module):
    """
    CNN-based binary image classifier.
    Supports ResNet and EfficientNet backbones with customizable classification head.
    """

    SUPPORTED_BACKBONES = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
        "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
    }

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the detector model.

        Args:
            backbone_name: Architecture backbone identifier (e.g. 'resnet50', 'efficientnet_b0').
            pretrained: If True, load ImageNet pretrained weights (transfer learning).
                        If False, train from scratch with random initialization.
            num_classes: Number of output classes (2 for Real vs AI-Generated).
            dropout_rate: Dropout probability in the classification head.
        """
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. "
                f"Supported: {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        model_fn, default_weights = self.SUPPORTED_BACKBONES[self.backbone_name]

        # Load backbone
        try:
            weights = default_weights if pretrained else None
            self.backbone = model_fn(weights=weights)
        except Exception:
            # Fallback for older torchvision compatibility
            self.backbone = model_fn(pretrained=pretrained)

        # Replace classification head
        if "resnet" in self.backbone_name:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._build_classifier_head(in_features, num_classes, dropout_rate)
        elif "efficientnet" in self.backbone_name:
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._build_classifier_head(in_features, num_classes, dropout_rate)

    @staticmethod
    def _build_classifier_head(
        in_features: int,
        num_classes: int,
        dropout_rate: float,
    ) -> nn.Sequential:
        """
        Build a multi-layer classification head with dropout and non-linearity.
        """
        hidden_dim = max(128, in_features // 2)
        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw class logits.
        """
        return self.backbone(x)


def build_model(config: Dict[str, Any]) -> AIImageDetector:
    """
    Instantiate and return the detector model based on configuration dictionary.

    Args:
        config: Loaded YAML configuration dictionary.

    Returns:
        AIImageDetector model instance.
    """
    model_cfg = config.get("model", {})
    backbone = model_cfg.get("backbone", "resnet50")
    pretrained = model_cfg.get("pretrained", True)
    num_classes = model_cfg.get("num_classes", 2)
    dropout_rate = model_cfg.get("dropout_rate", 0.3)

    model = AIImageDetector(
        backbone_name=backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    return model
