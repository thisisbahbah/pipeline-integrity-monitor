"""
gradcam.py
----------
GradCAM implementation from scratch using PyTorch forward/backward hooks.
NO third-party GradCAM library — hooks are registered directly on the target layer.

GradCAM reference: Selvaraju et al., 2017 (https://arxiv.org/abs/1610.02391)

How it works:
  1. Forward pass: register a hook to capture the feature map activations
     from the target conv layer.
  2. Backward pass: register a hook to capture the gradients of the loss
     w.r.t. those feature maps.
  3. Global-average-pool the gradients across spatial dimensions → weights α_k
  4. Weighted sum of feature maps: L_GradCAM = ReLU(Σ α_k * A_k)
  5. Upsample to input resolution and overlay on the original image.

Usage:
    from scripts.gradcam import GradCAM, overlay_heatmap

    cam = GradCAM(model, target_layer=model.layer4[-1])  # ResNet-18 example
    heatmap = cam.generate(input_tensor, class_idx=None)  # None → argmax
    overlay = overlay_heatmap(original_pil_image, heatmap)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional


class GradCAM:
    """
    GradCAM for any PyTorch classification model.

    Args:
        model:        Trained PyTorch model in eval mode.
        target_layer: The convolutional layer to hook.
                      ResNet-18:       model.layer4[-1]
                      EfficientNet-B0: model.features[-1]
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _save_activations(self, module, input, output):
        # output shape: (batch, C, H, W)
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0] shape: (batch, C, H, W)
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a GradCAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor, shape (1, 3, H, W).
            class_idx:    Target class index. None → use argmax of model output.

        Returns:
            heatmap: np.ndarray of shape (H, W), values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        logits = self.model(input_tensor)           # (1, num_classes)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward — zero out everything except the target class score
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # α_k = global-average-pool over spatial dims of the gradient
        # gradients: (1, C, h, w)  →  weights: (C,)
        assert self._gradients is not None, "Gradients not captured — did backward() run?"
        assert self._activations is not None, "Activations not captured — did forward() run?"
        gradients   = self._gradients[0]    # (C, h, w)
        activations = self._activations[0]  # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of forward activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for k, w in enumerate(weights):
            cam += w * activations[k]

        # ReLU — we only care about features that positively influence the class
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam_np = cam.numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np  # (h_feat, w_feat)

    def remove_hooks(self):
        """Call when done — prevents memory leaks from dangling hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ---------------------------------------------------------------------------
# Overlay utility
# ---------------------------------------------------------------------------

def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay a GradCAM heatmap onto the original PIL image.

    Args:
        original_image: PIL Image (RGB), any size.
        heatmap:        np.ndarray in [0, 1], output of GradCAM.generate().
        alpha:          Heatmap opacity (0 = invisible, 1 = fully opaque).
        colormap:       Matplotlib colormap name (default: 'jet').

    Returns:
        Blended PIL Image (RGB).
    """
    import matplotlib.cm as cm  # lazy import — only needed here

    # Upscale heatmap to match image
    h, w = original_image.size[1], original_image.size[0]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), Image.Resampling.BILINEAR
        )
    ) / 255.0

    # Apply colormap → RGBA → RGB
    cmap = cm.get_cmap(colormap)
    coloured = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    coloured_pil = Image.fromarray(coloured)

    # Blend
    original_np = np.array(original_image.convert("RGB"), dtype=np.float32)
    coloured_np = np.array(coloured_pil, dtype=np.float32)
    blended = ((1 - alpha) * original_np + alpha * coloured_np).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Target-layer helper
# ---------------------------------------------------------------------------

def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """
    Return the canonical GradCAM target layer for supported architectures.
    Modify this if you add new backbones.
    """
    import torch.nn as nn
    from torchvision.models import ResNet, EfficientNet

    if model_name == "resnet18":
        assert isinstance(model, ResNet)
        layer: nn.Module = model.layer4[-1]
        return layer
    elif model_name == "efficientnet_b0":
        assert isinstance(model, EfficientNet)
        layer = model.features[-1]
        return layer
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Add its target layer to get_target_layer()."
        )