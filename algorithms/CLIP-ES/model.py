"""
model.py 定义了 CLIP-ES 算法 (CLIP is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation) 的网络架构

    @Time    : 2025/05/19
    @Author  : JackWang
    @File    : model.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library

# Third-Party Library
import torch
import torch.nn as nn

# Torch Library

# My Library


class CLIPESModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(768, 768)
