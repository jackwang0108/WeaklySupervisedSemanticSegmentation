"""
base.py 定义了算法的抽象基类, 所有算法都需要继承这个类

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : base.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from typing import Any, Literal
from abc import ABC, abstractmethod

# Third-Party Library
from omegaconf import DictConfig

# Torch Library
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# My Library
from core.logger import RichuruLogger


class WeaklySupervisedSemanticSegmentationAlgorithm(ABC):
    """弱监督语义分割算法的抽象基类"""

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

        self.model = self.build_model()
        self.logger: RichuruLogger = None
        self.writer: SummaryWriter = None

        self.info_dict: dict[str, Any] = None

    @abstractmethod
    def build_model(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self, data: list[torch.Tensor], epoch: int, batch: int, num_batches: int
    ) -> dict[Literal["loss"] | str, torch.Tensor | Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image):
        """返回分割结果, 形状必须是 [B, H, W]"""
        raise NotImplementedError

    @abstractmethod
    def get_info_dict(self) -> dict[str, Any]:
        raise NotImplementedError
