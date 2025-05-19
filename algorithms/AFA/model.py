"""
model.py 定义了 AFA 算法 (Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers) 的网络架构

    @Time    : 2025/05/19
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


class AFAModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(768, 768)
