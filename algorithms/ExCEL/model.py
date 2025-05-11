"""
model.py 定义了ExCEL算法 (Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation) 的网络架构

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


class ExCELModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(512, 512)
