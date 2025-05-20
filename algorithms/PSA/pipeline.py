"""
pipeline.py 定义了 PSA算法 (Learning Pixel-level Semantic Affinity with Image-level Supervision) 的训练流程

    @Time    : 2025/05/20
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


class PSA(WSSSAlgorithm):

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def build_model(self):
        return super().build_model()

    def train_step(self, data, epoch, batch, num_batches):
        return super().train_step(data, epoch, batch, num_batches)

    def predict(self, image):
        return super().predict(image)

    def get_info_dict(self):
        return super().get_info_dict()
