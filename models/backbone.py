"""MobileNetV3-Large backbone for feature extraction.

Loads ImageNet-pretrained MobileNetV3-Large, strips the classifier,
and exposes the convolutional feature extractor. Output: 960 channels
at 1/32 of input spatial resolution (256x192 -> 8x6).

Supports 6-channel input (RGB + classical CV channels) by replacing
the first conv layer and copying pretrained weights for the RGB channels.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV3Backbone(nn.Module):

    def __init__(self, pretrained=True, freeze_early=True, in_channels=3):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)

        self.features = mobilenet.features
        self.out_channels = 960

        # Replace first conv if input is not 3 channels
        if in_channels != 3:
            old_conv = self.features[0][0]  # First conv in the first block
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=old_conv.bias is not None,
            )
            # Copy pretrained weights for first 3 channels, init rest with small random
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode='fan_out')
            self.features[0][0] = new_conv

        if freeze_early:
            for i, layer in enumerate(self.features):
                if i < 4:
                    for param in layer.parameters():
                        param.requires_grad = False

    def unfreeze_all(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.features(x)
