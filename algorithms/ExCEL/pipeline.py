"""
pipeline.py 定义了ExCEL算法 (Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation) 的训练流程

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : pipeline.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library

# Third-Party Library
from omegaconf import DictConfig

# Torch Library

# My Library
from .. import register_algorithm
from ..base import WeaklySupervisedSemanticSegmentationAlgorithm as WSSSAlgorithm

from .model import ExCELModel
from .helper import ClassNames, get_descriptions


@register_algorithm("ExCEL")
class ExCEL(WSSSAlgorithm):
    def __init__(self, config: DictConfig):
        super().__init__(config=config)

    def build_model(self) -> ExCELModel:
        print("Building ExCEL model...")
        return ExCELModel()

    def train_step(self):
        print("Training ExCEL model...")

    def predict(self, image):
        print("Predicting with ExCEL model...")
        return image
