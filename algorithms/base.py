"""
base.py 定义了算法的抽象基类, 所有算法都需要继承这个类

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : base.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from abc import ABC, abstractmethod

# Third-Party Library

# Torch Library
import torch.nn as nn

# My Library


class WeaklySupervisedSemanticSegmentationAlgorithm(ABC):
    """弱监督语义分割算法的抽象基类"""

    @abstractmethod
    def build_model(self, cfg) -> nn.Module:
        """构建模型"""
        pass

    @abstractmethod
    def train_step(self, data, epoch, step) -> dict:
        """训练一步"""
        pass

    @abstractmethod
    def predict(self, image):
        """推理接口"""
        pass
