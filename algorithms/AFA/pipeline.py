"""
pipeline.py 定义了 AFA 算法 (Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers) 的训练流程

    @Time    : 2025/05/19
    @Author  : JackWang
    @File    : pipeline.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library

# Third-Party Library
from omegaconf import DictConfig

# Torch Library
import torch
import torch.nn.functional as F

# My Library
from .. import register_algorithm
from ..base import WeaklySupervisedSemanticSegmentationAlgorithm as WSSSAlgorithm

from .model import AFAModel


@register_algorithm("AFA")
class AFA(WSSSAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.info_dict = {}

    def build_model(self) -> AFAModel:
        return super().build_model()

    def train_step(
        self, data: list[torch.Tensor], epoch: int, batch: int, num_batches: int
    ) -> dict[str, torch.Tensor]:
        return super().train_step(data, epoch, batch, num_batches)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return super().predict(image)

    def get_info_dict(self) -> dict[str, str]:
        return super().get_info_dict()
