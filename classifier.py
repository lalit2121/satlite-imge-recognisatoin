"""
===========================================================================
  SATELLITE IMAGE CLASSIFIER
  Transfer Learning with EfficientNet-B0 / ResNet50 / ViT-Tiny
===========================================================================

WHAT IS TRANSFER LEARNING?
---------------------------
Instead of training a neural network from scratch (which requires
millions of images and days of GPU time), we start from a model that
was already trained on ImageNet — a dataset of 1.2 million photos.

The key insight: the low-level features learned from natural images
(edges, textures, colour gradients) are also useful for satellite images.
Only the final classification layer needs to be re-trained from scratch.

This is called "fine-tuning":
  1. Load pretrained weights (backbone frozen)
  2. Replace final layer for our 10 classes
  3. First train only the new head (few epochs)
  4. Then unfreeze backbone and train everything at low learning rate

WHY EFFICIENTNET-B0?
--------------------
EfficientNet was designed with compound scaling — systematically
scaling depth, width, and resolution together. EfficientNet-B0 is:
  - 5.3M parameters (vs 25M for ResNet50)
  - ~3× fewer operations than ResNet50
  - Top-1 ImageNet accuracy: 77.1% (ResNet50: 76.1%)
  - CPU training time: ~2-3 hours for 30 epochs on EuroSAT
  - Memory: ~2 GB RAM for batch_size=32

WHY NOT ViT (Vision Transformer)?
----------------------------------
Full ViT requires large datasets (ImageNet-21k) to pretrain well.
ViT-Tiny is feasible but EfficientNet-B0 outperforms it on small
datasets like EuroSAT. We provide ViT-Tiny as an optional experiment.

MODEL ARCHITECTURE:
-------------------
Input (3×224×224)
  ↓
EfficientNet-B0 backbone (frozen during warmup)
  ↓
Adaptive Average Pooling → (1280,)
  ↓
Dropout(0.3)
  ↓
Linear(1280 → 512) → BatchNorm → ReLU
  ↓
Dropout(0.2)
  ↓
Linear(512 → 10) → log_softmax
  ↓
Output: class probabilities for 10 land-cover classes
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Tuple

import torchvision.models as tv_models


# ─── Available architectures ─────────────────────────────────────────────────
SUPPORTED_MODELS = ["efficientnet_b0", "resnet50", "vit_tiny", "mobilenet_v3"]


class SatelliteClassifier(nn.Module):
    """
    Transfer learning classifier for EuroSAT satellite images.

    The model has two phases:
      Phase 1 — HEAD ONLY training:
        Backbone weights are frozen (not updated).
        Only the new classification head is trained.
        Use this for the first 5–10 epochs.

      Phase 2 — FULL FINE-TUNING:
        All weights are unfrozen and updated.
        Use a low learning rate (1e-4 or lower) to avoid destroying
        the pretrained features.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 10,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        """
        Args:
            model_name:  Architecture to use (see SUPPORTED_MODELS)
            num_classes: Number of output classes (10 for EuroSAT)
            dropout:     Dropout probability in the classification head
            pretrained:  Load ImageNet pretrained weights
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout

        # Build backbone + head
        self.backbone, self.head, self.feature_dim = self._build(
            model_name, num_classes, dropout, pretrained
        )

        # Freeze backbone initially (Phase 1 training)
        self.freeze_backbone()

        # Track training phase
        self.backbone_frozen = True

    # ─────────────────────────────────────────────────────────────────────────
    #  MODEL CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────
    def _build(
        self, model_name: str, num_classes: int, dropout: float, pretrained: bool
    ) -> Tuple[nn.Module, nn.Module, int]:
        """
        Load pretrained backbone and attach a custom classification head.

        For each architecture, we:
          1. Load the torchvision pretrained model
          2. REMOVE the original classification layer
          3. REPLACE with our multi-class head for EuroSAT

        Returns: (backbone, head, feature_dim)
        """
        weights_arg = "DEFAULT" if pretrained else None

        if model_name == "efficientnet_b0":
            # EfficientNet-B0: 5.3M params, very CPU-friendly
            base = tv_models.efficientnet_b0(weights=weights_arg)
            feature_dim = base.classifier[1].in_features   # 1280

            # Remove original classifier
            backbone = base.features  # All convolutional layers
            backbone.add_module("pool", nn.AdaptiveAvgPool2d(1))

            head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.7),
                nn.Linear(512, num_classes),
            )

        elif model_name == "resnet50":
            # ResNet50: 25.6M params, strong baseline, moderate CPU time
            base = tv_models.resnet50(weights=weights_arg)
            feature_dim = base.fc.in_features   # 2048

            # Remove the fully connected layer
            backbone = nn.Sequential(*list(base.children())[:-1])  # All except fc

            head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.7),
                nn.Linear(512, num_classes),
            )

        elif model_name == "mobilenet_v3":
            # MobileNetV3-Small: 2.5M params, fastest on CPU
            base = tv_models.mobilenet_v3_small(weights=weights_arg)
            feature_dim = 576

            backbone = base.features
            backbone.add_module("pool", nn.AdaptiveAvgPool2d(1))

            head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_classes),
            )

        elif model_name == "vit_tiny":
            # ViT-Tiny via timm (if available), else fallback to EfficientNet
            try:
                import timm
                base = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
                feature_dim = base.embed_dim   # 192

                backbone = base

                head = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, num_classes),
                )
            except ImportError:
                print("timm not installed — falling back to EfficientNet-B0")
                return self._build("efficientnet_b0", num_classes, dropout, pretrained)

        else:
            raise ValueError(f"Unknown model: {model_name}. Choose from {SUPPORTED_MODELS}")

        return backbone, head, feature_dim

    # ─────────────────────────────────────────────────────────────────────────
    #  FORWARD PASS
    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → class logits.

        The backbone extracts spatial features from the satellite image.
        The head maps those features to class scores.

        Args:
            x: Image tensor of shape [batch_size, 3, 224, 224]
        Returns:
            logits: Shape [batch_size, num_classes] (raw scores, not probabilities)
        """
        if self.model_name == "vit_tiny":
            # ViT returns CLS token directly
            features = self.backbone(x)      # [B, feature_dim]
        else:
            features = self.backbone(x)      # [B, feature_dim, 1, 1] or [B, feature_dim]

        logits = self.head(features)         # [B, num_classes]
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax → class probabilities [0, 1]."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → predicted class indices."""
        return self.forward(x).argmax(dim=1)

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE CONTROL: Freeze / Unfreeze backbone
    # ─────────────────────────────────────────────────────────────────────────
    def freeze_backbone(self):
        """
        Freeze all backbone parameters.
        Only the classification head will be trained.
        Use during the warmup phase (first 5–10 epochs).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True

    def unfreeze_backbone(self, unfreeze_last_n_layers: Optional[int] = None):
        """
        Unfreeze backbone for full fine-tuning.

        Args:
            unfreeze_last_n_layers: If set, only unfreeze the last N layers
                                    (useful when memory/compute is limited).
                                    If None, unfreeze everything.

        WHY UNFREEZE GRADUALLY?
        The early layers (edges, colours) are very general — they don't need
        much updating. The later layers (complex patterns) are more task-specific
        and benefit from fine-tuning. Unfreezing everything at once with a high
        LR can "catastrophically forget" the pretrained features.
        """
        if unfreeze_last_n_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze only the last N children of backbone
            all_children = list(self.backbone.children())
            for child in all_children[-unfreeze_last_n_layers:]:
                for param in child.parameters():
                    param.requires_grad = True

        self.backbone_frozen = False

    def get_trainable_params(self) -> int:
        """Count trainable (unfrozen) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count all parameters."""
        return sum(p.numel() for p in self.parameters())

    def param_summary(self):
        """Print model parameter summary."""
        total = self.get_total_params()
        trainable = self.get_trainable_params()
        frozen = total - trainable
        print(f"\n{'='*50}")
        print(f"  Model: {self.model_name}")
        print(f"{'='*50}")
        print(f"  Total parameters:     {total:>12,}")
        print(f"  Trainable parameters: {trainable:>12,}  ({trainable/total*100:.1f}%)")
        print(f"  Frozen parameters:    {frozen:>12,}  ({frozen/total*100:.1f}%)")
        print(f"  Backbone frozen:      {self.backbone_frozen}")
        print(f"{'='*50}\n")

    # ─────────────────────────────────────────────────────────────────────────
    #  FEATURE EXTRACTION (for Grad-CAM and analysis)
    # ─────────────────────────────────────────────────────────────────────────
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps from backbone (used by Grad-CAM)."""
        with torch.no_grad():
            features = self.backbone(x)
        return features

    def get_last_conv_layer(self) -> nn.Module:
        """Return the last convolutional layer (target for Grad-CAM)."""
        if self.model_name == "efficientnet_b0":
            # Last block of EfficientNet features
            return self.backbone[-2]
        elif self.model_name == "resnet50":
            return list(self.backbone.children())[-3][-1]
        else:
            return list(self.backbone.modules())[-1]


# ─── Factory functions ────────────────────────────────────────────────────────
def build_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 10,
    dropout: float = 0.3,
    pretrained: bool = True,
    device: Optional[str] = None,
) -> SatelliteClassifier:
    """
    Build and return a SatelliteClassifier on the specified device.

    Args:
        model_name:  Architecture (efficientnet_b0, resnet50, mobilenet_v3, vit_tiny)
        num_classes: 10 for EuroSAT
        dropout:     Dropout rate (0.3 is a good default)
        pretrained:  Load ImageNet weights
        device:      "cpu", "cuda", or None (auto-detect)

    Returns:
        model (SatelliteClassifier) on device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SatelliteClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )
    model = model.to(device)

    print(f"Model '{model_name}' built on device: {device}")
    model.param_summary()

    return model


def save_model(
    model: SatelliteClassifier,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    val_acc: float = 0.0,
    extra: Optional[Dict] = None,
):
    """
    Save model checkpoint with all metadata needed for resuming training.

    We save a CHECKPOINT (not just weights) so training can be resumed:
      - model state dict (all weights)
      - optimizer state dict (momentum, etc.)
      - epoch number
      - best validation accuracy
      - any extra metadata
    """
    checkpoint = {
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "dropout": model.dropout,
        "backbone_frozen": model.backbone_frozen,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        checkpoint.update(extra)

    os.makedirs(Path(save_path).parent, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")


def load_model(
    load_path: str,
    device: Optional[str] = None,
) -> Tuple[SatelliteClassifier, Dict]:
    """
    Load a saved model checkpoint.

    Returns:
        (model, checkpoint_dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(load_path, map_location=device)

    model = SatelliteClassifier(
        model_name=checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint.get("dropout", 0.3),
        pretrained=False,  # Weights come from checkpoint
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from: {load_path}")
    print(f"  Architecture: {checkpoint['model_name']}")
    print(f"  Epoch:        {checkpoint.get('epoch', '?')}")
    print(f"  Val accuracy: {checkpoint.get('val_acc', 0):.4f}")

    return model, checkpoint


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Satellite Classifier — Model Test")
    print("=" * 60)

    for arch in ["efficientnet_b0", "resnet50", "mobilenet_v3"]:
        print(f"\nTesting {arch}...")
        model = build_model(arch, num_classes=10)

        # Test forward pass
        dummy = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)

        print(f"  Input:  {dummy.shape}")
        print(f"  Output: {out.shape}")    # Should be [4, 10]
        assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"

        # Unfreeze and check trainable params change
        model.unfreeze_backbone()
        model.param_summary()

    print("\nAll model tests passed!")
