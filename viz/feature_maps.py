"""CNN feature map visualization.

Extracts and renders intermediate activation maps from the backbone.
Early layers show edge/gradient responses (like Sobel/Canny).
Deep layers show semantic body-part activations.
"""

import numpy as np
import cv2
import torch


def extract_feature_maps(model, image_tensor):
    """Extract early and deep feature maps from the backbone.

    Args:
        model: GymPoseLite model
        image_tensor: (1, 3, 256, 192) input tensor

    Returns:
        early_maps: (C_early, H, W) activations from layer 1
        deep_maps: (C_deep, H, W) activations from last layer
    """
    model.eval()
    with torch.no_grad():
        early, deep = model.get_feature_maps(image_tensor)
    # Remove batch dim
    return early[0].cpu().numpy(), deep[0].cpu().numpy()


def select_top_channels(feature_maps, n=4):
    """Select the N channels with highest mean activation.

    These are the most "active" filters for the current input.
    """
    # Mean activation per channel
    mean_act = feature_maps.mean(axis=(1, 2))  # (C,)
    top_indices = np.argsort(mean_act)[-n:][::-1]
    return feature_maps[top_indices], top_indices


def render_feature_grid(feature_maps, n=4, panel_size=(200, 200)):
    """Render top-N feature maps as a 2x2 grid image.

    Each map is normalized to [0, 255] and colormapped with JET
    for clear visualization of activation patterns.
    """
    maps, indices = select_top_channels(feature_maps, n)

    panels = []
    for i in range(n):
        fm = maps[i]
        # Normalize to 0-255
        fm = fm - fm.min()
        if fm.max() > 0:
            fm = fm / fm.max()
        fm = (fm * 255).astype(np.uint8)
        # Resize and apply colormap
        fm = cv2.resize(fm, panel_size, interpolation=cv2.INTER_LINEAR)
        fm = cv2.applyColorMap(fm, cv2.COLORMAP_JET)
        panels.append(fm)

    # Arrange as 2x2 grid
    row1 = np.hstack(panels[:2])
    row2 = np.hstack(panels[2:4])
    grid = np.vstack([row1, row2])
    return grid
