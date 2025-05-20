"""
model.py 定义了 PSA算法 (Learning Pixel-level Semantic Affinity with Image-level Supervision) 的网络架构

    @Time    : 2025/05/20
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


class PSAModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(512, 512)
