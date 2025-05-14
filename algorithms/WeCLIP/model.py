"""
model.py 定义了WeCLIP算法 (Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation) 的网络架构

@Time    : 2025/05/14
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


class WeCLIPModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
