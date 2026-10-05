"""Pose head: heatmap generation + soft-argmax coordinate extraction.

Takes 256-channel feature map at 64x48 and produces:
- 17 heatmaps (one per COCO keypoint) — spatial probability maps
- 17x2 (x,y) coordinates via soft-argmax — sub-pixel precision

The soft-argmax is differentiable (unlike hard argmax), enabling
end-to-end training with coordinate supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseHead(nn.Module):

    def __init__(self, in_channels=256, num_keypoints=17, beta=100.0):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.beta = beta  # Soft-argmax temperature (higher = sharper peak)

        # 1x1 conv to project features -> one heatmap per keypoint
        self.heatmap_conv = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

    def soft_argmax(self, heatmaps):
        """Extract (x, y) coordinates from heatmaps using spatial soft-argmax.

        For each heatmap:
        1. Multiply by temperature beta to sharpen the distribution
        2. Apply spatial softmax -> probability distribution over pixels
        3. Compute expected x = sum(p * x_grid), expected y = sum(p * y_grid)
        This gives sub-pixel precision and is fully differentiable.
        """
        B, K, H, W = heatmaps.shape

        # Create coordinate grids [0, 1] range
        device = heatmaps.device
        y_grid = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, K, H, W)
        x_grid = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, K, H, W)

        # Spatial softmax: flatten spatial dims, softmax, reshape back
        flat = heatmaps.view(B, K, -1)
        weights = F.softmax(flat * self.beta, dim=-1).view(B, K, H, W)

        # Expected coordinates (weighted sum)
        x = (weights * x_grid).sum(dim=(2, 3))  # (B, K)
        y = (weights * y_grid).sum(dim=(2, 3))  # (B, K)

        coords = torch.stack([x, y], dim=-1)  # (B, K, 2)
        return coords

    def forward(self, features):
        raw_heatmaps = self.heatmap_conv(features)  # (B, 17, 64, 48)
        heatmaps = torch.sigmoid(raw_heatmaps)      # Bound to [0, 1] to match GT range
        coords = self.soft_argmax(heatmaps)          # (B, 17, 2)
        return heatmaps, coords
