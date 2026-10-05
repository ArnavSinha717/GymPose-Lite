"""Heatmap visualization: grid display and overlay on original image.

Shows the 17 spatial probability maps (one per keypoint) that are
the core output of the pose estimation network. Each heatmap peaks
at the predicted joint location.
"""

import numpy as np
import cv2

JOINT_NAMES = [
    "nose", "L eye", "R eye", "L ear", "R ear",
    "L shoulder", "R shoulder", "L elbow", "R elbow",
    "L wrist", "R wrist", "L hip", "R hip",
    "L knee", "R knee", "L ankle", "R ankle",
]


def render_heatmap_grid(heatmaps, cell_size=80):
    """Render 17 heatmaps as a labeled grid.

    Layout: 3 rows x 6 cols (last cell blank).
    Each cell shows one joint's heatmap with its name.
    """
    rows, cols = 3, 6
    pad = 2
    grid_h = rows * (cell_size + pad) + pad
    grid_w = cols * (cell_size + pad) + pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for k in range(17):
        r, c = divmod(k, cols)
        y = pad + r * (cell_size + pad)
        x = pad + c * (cell_size + pad)

        # Normalize and colormap
        hm = heatmaps[k]
        hm = hm - hm.min()
        if hm.max() > 0:
            hm = hm / hm.max()
        hm = (hm * 255).astype(np.uint8)
        hm = cv2.resize(hm, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_HOT)

        # Place in grid
        grid[y:y + cell_size, x:x + cell_size] = hm_color

        # Label
        cv2.putText(grid, JOINT_NAMES[k], (x + 2, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return grid


def render_heatmap_overlay(image, heatmaps, alpha=0.5):
    """Overlay summed heatmaps on original image as a colormap blend.

    Sums all 17 heatmaps, colormaps the result, and blends
    with the original image for a visual heat overlay.
    """
    h, w = image.shape[:2]

    # Sum all heatmaps
    combined = heatmaps.sum(axis=0)
    combined = combined - combined.min()
    if combined.max() > 0:
        combined = combined / combined.max()
    combined = (combined * 255).astype(np.uint8)
    combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap(combined, cv2.COLORMAP_JET)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
