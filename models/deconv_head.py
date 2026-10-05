"""Deconvolution (transposed convolution) head for spatial upsampling.

Recovers spatial resolution lost during backbone feature extraction.
Uses standard transposed convolutions for better precision.
960 channels at 8x6 -> 256 channels at 64x48 (3 layers, each 2x upsample).
"""

import torch
import torch.nn as nn


class DeconvBlock(nn.Module):
    """Standard transposed convolution + BatchNorm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeconvHead(nn.Module):

    def __init__(self, in_channels=960, out_channels=256, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            inc = in_channels if i == 0 else out_channels
            layers.append(DeconvBlock(inc, out_channels))
        self.deconv = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.deconv(x)
