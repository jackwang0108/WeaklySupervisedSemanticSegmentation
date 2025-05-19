"""
pipeline.py 定义了 CLIP-ES 算法 (CLIP is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation) 的训练流程

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

# My Library
from .. import register_algorithm
from ..base import WeaklySupervisedSemanticSegmentationAlgorithm as WSSSAlgorithm


@register_algorithm("CLIP-ES")
class CLIPES(WSSSAlgorithm):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.info_dict = {}

    def build_model(self):
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
