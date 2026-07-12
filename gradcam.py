"""
===========================================================================
  GRAD-CAM: Gradient-weighted Class Activation Mapping
  Visualise WHERE the model looks to make its decision
===========================================================================

WHAT IS GRAD-CAM?
------------------
When a CNN classifies "Forest", which pixels did it use?
Grad-CAM answers this by computing a heatmap over the input image
showing which spatial locations most influenced the prediction.

HOW IT WORKS (intuitively):
----------------------------
1. Do a forward pass: image → features → prediction
2. Compute gradient of the class score with respect to the final
   convolutional feature maps.
3. Weights that have a large positive gradient → that spatial region
   was IMPORTANT for this prediction.
4. Average the gradients over channels → one weight per feature map.
5. Take weighted sum of feature maps → raw attention map.
6. Resize to input image size and overlay as a heatmap.

MATHEMATICS:
------------
For class c, the Grad-CAM activation map is:

  L^c_GradCAM = ReLU( Σ_k  α_k^c · A^k )

  where:
    A^k = k-th feature map of the last conv layer
    α_k^c = (1/Z) Σ_ij ∂y^c / ∂A^k_ij   (global average of gradients)

  ReLU removes negative activations (we only care about features that
  increase the score for class c, not decrease it).

WHY IS THIS USEFUL?
--------------------
  1. Debugging: if the heatmap highlights background instead of object → problem!
  2. Trust: users can verify the model is using the right features
  3. Science: reveals what visual patterns distinguish land-cover classes
  4. Portfolio: very visually impressive in README and presentations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation using PyTorch forward and backward hooks.

    Usage:
        gradcam = GradCAM(model, target_layer=model.backbone[-2])
        heatmap, pred_class = gradcam(image_tensor)
        gradcam.overlay(image_pil, heatmap, save_path="gradcam_output.png")
        gradcam.remove_hooks()  # Always call this when done!
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model:        Trained classifier
            target_layer: The layer whose activations we visualise.
                         Usually the last convolutional layer.
                         For EfficientNet-B0: model.backbone[-2]
                         For ResNet50:        model.backbone[-3][-1]
        """
        self.model = model
        self.target_layer = target_layer

        # Storage for activations and gradients
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks (called automatically during forward/backward)
        self._forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        """Hook: save feature map activations during forward pass."""
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Hook: save gradients during backward pass."""
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM heatmap for an input image.

        Args:
            image:        Input tensor [1, 3, H, W] (single image)
            target_class: Class index to visualise.
                         If None, uses the predicted (top-1) class.

        Returns:
            (heatmap, predicted_class, confidence)
            heatmap: np.ndarray of shape [H, W], values in [0, 1]
        """
        self.model.eval()
        image.requires_grad_(False)

        # Forward pass
        logits = self.model(image)
        probs  = F.softmax(logits, dim=1)

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        confidence = probs[0, target_class].item()

        # Backward pass for the target class
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()

        # Grad-CAM computation
        # _activations: [1, C, h, w]
        # _gradients:   [1, C, h, w]
        activations = self._activations.squeeze(0)  # [C, h, w]
        gradients   = self._gradients.squeeze(0)    # [C, h, w]

        # Global average pool of gradients → weight per channel
        weights = gradients.mean(dim=(1, 2))        # [C]

        # Weighted sum of activation maps
        cam = (weights[:, None, None] * activations).sum(dim=0)  # [h, w]

        # ReLU: keep only positive contributions
        cam = F.relu(cam)

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        h, w = image.shape[2], image.shape[3]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        return cam, target_class, confidence

    def overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap onto the original image.

        Args:
            original_image: np.ndarray [H, W, 3], values in [0, 1]
            heatmap:        np.ndarray [H, W], values in [0, 1]
            alpha:          Heatmap opacity (0 = invisible, 1 = opaque)
            colormap:       Matplotlib colormap name (e.g., 'jet', 'hot', 'plasma')
            save_path:      If set, save figure to this path

        Returns:
            Overlaid image as np.ndarray [H, W, 3]
        """
        # Convert heatmap to RGB using colormap
        cmap = plt.get_cmap(colormap)
        heatmap_rgb = cmap(heatmap)[:, :, :3]   # [H, W, 3], drop alpha channel

        # Blend with original image
        overlay = (1 - alpha) * original_image + alpha * heatmap_rgb
        overlay = np.clip(overlay, 0, 1)

        if save_path is not None:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("Original Image", fontsize=11)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, cmap=colormap)
            plt.title("Grad-CAM Heatmap", fontsize=11)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("Overlay", fontsize=11)
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        return overlay

    def remove_hooks(self):
        """Always call this after using GradCAM to free memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ─── Batch Grad-CAM visualization ────────────────────────────────────────────
def plot_gradcam_grid(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    target_layer: nn.Module,
    device: torch.device,
    n_images: int = 8,
    save_path: Optional[str] = None,
):
    """
    Plot a grid of Grad-CAM overlays for multiple images.

    Each row shows: Original | Grad-CAM | Overlay
    with true label vs predicted label and confidence.
    """
    from eurosat_dataset import denormalize

    gradcam = GradCAM(model, target_layer)
    n_images = min(n_images, len(images))

    fig, axes = plt.subplots(n_images, 3, figsize=(12, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_images):
        img_tensor = images[i:i+1].to(device)
        true_class = labels[i].item()

        # Denormalize for display
        img_display = denormalize(images[i]).permute(1, 2, 0).numpy()

        # Compute Grad-CAM
        heatmap, pred_class, confidence = gradcam(img_tensor)

        # Build colormap overlay
        cmap = plt.get_cmap("jet")
        heatmap_rgb = cmap(heatmap)[:, :, :3]
        overlay = 0.5 * img_display + 0.5 * heatmap_rgb
        overlay = np.clip(overlay, 0, 1)

        # Plot
        correct = "✓" if pred_class == true_class else "✗"
        color = "green" if pred_class == true_class else "red"

        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"True: {class_names[true_class]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f"{correct} Pred: {class_names[pred_class]} ({confidence*100:.1f}%)",
            fontsize=9, color=color
        )
        axes[i, 2].axis("off")

    plt.suptitle("Grad-CAM: Model Attention Visualisation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM grid saved to: {save_path}")
    plt.close()

    gradcam.remove_hooks()
