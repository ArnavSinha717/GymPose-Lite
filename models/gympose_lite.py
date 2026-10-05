"""GymPose-Lite: full model combining backbone, deconv, pose head, and GNN.

Pipeline: Image -> MobileNetV3 -> Deconv Upsample -> Heatmaps -> Soft-argmax -> GNN Refine
"""

import torch
import torch.nn as nn

from .backbone import MobileNetV3Backbone
from .deconv_head import DeconvHead
from .pose_head import PoseHead
from .gnn_refine import GNNRefine


class GymPoseLite(nn.Module):

    def __init__(self, num_keypoints=17, use_gnn=True, gnn_hidden=64, pretrained=True, in_channels=3):
        super().__init__()
        self.use_gnn = use_gnn

        self.backbone = MobileNetV3Backbone(pretrained=pretrained, in_channels=in_channels)
        self.deconv = DeconvHead(
            in_channels=self.backbone.out_channels,
            out_channels=256,
            num_layers=3,
        )
        self.pose_head = PoseHead(
            in_channels=256,
            num_keypoints=num_keypoints,
        )
        if use_gnn:
            self.gnn = GNNRefine(
                num_joints=num_keypoints,
                hidden_dim=gnn_hidden,
            )

    def forward(self, x, exercise="general"):
        """
        x: (B, 3, 256, 192) input image
        Returns:
            heatmaps: (B, 17, 64, 48) spatial probability maps
            coords: (B, 17, 2) keypoint coordinates in [0, 1]
        """
        features = self.backbone(x)          # (B, 576, 8, 6)
        upsampled = self.deconv(features)    # (B, 256, 64, 48)
        heatmaps, coords = self.pose_head(upsampled)  # (B,17,64,48), (B,17,2)

        if self.use_gnn:
            coords = self.gnn(coords, exercise)

        return heatmaps, coords

    def get_feature_maps(self, x):
        """Extract intermediate feature maps for visualization.

        Returns early (edge/gradient) and deep (semantic) activation maps.
        """
        early_features = None
        deep_features = None

        # Run through backbone blocks, capture early and late
        h = x
        for i, block in enumerate(self.backbone.features):
            h = block(h)
            if i == 1:
                early_features = h.detach()
            if i == len(self.backbone.features) - 1:
                deep_features = h.detach()

        return early_features, deep_features
