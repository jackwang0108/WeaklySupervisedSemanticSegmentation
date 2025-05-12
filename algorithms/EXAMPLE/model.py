"""
model.py 定义了EXAMPLE算法的模型

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : model.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn

# My Library


class ExampleModel(nn.Module):
    def __init__(
        self, in_channels: int = 3, out_channels: int = 256, num_classes: int = 21
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.conv(image)
        feature = self.avg_pool(feature)
        feature = self.flatten(feature)
        return self.fc(feature)
