"""
pipeline.py 定义了ExCEL算法 (Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation) 的训练流程

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : pipeline.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import random
from typing import Any

# Third-Party Library
from omegaconf import DictConfig

# Torch Library
import torch

# My Library
from .. import register_algorithm
from ..base import WeaklySupervisedSemanticSegmentationAlgorithm as WSSSAlgorithm

from .model import ExampleModel


@register_algorithm("EXAMPLE")
class EXAMPLE(WSSSAlgorithm):
    def __init__(self, config: DictConfig):
        super().__init__(config=config)

        self.info_dict = {}

    def build_model(self) -> ExampleModel:
        return ExampleModel(
            self.config.algorithm.in_channels,
            self.config.algorithm.out_channels,
            self.config.algorithm.num_classes,
        )

    def train_step(
        self, data: tuple[torch.Tensor], epoch: int, batch: int
    ) -> dict[str, Any]:

        preds: torch.Tensor = self.model(data[0])

        loss = preds.mean(dim=0).sum().abs()

        self.info_dict |= {"epoch": epoch, "batch": batch, "loss": loss}

        return {"loss": loss}

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def get_info_dict(self):
        return self.info_dict
