"""


@Time    : 2025/05/14
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

from .model import WeCLIPModel

# from .helper import ClassNames, get_descriptions


@register_algorithm("WeCLIP")
class WeCLIP(WSSSAlgorithm):
    def __init__(self, config: DictConfig):
        super().__init__(config=config)

    def build_model(self) -> WeCLIPModel:
        print("Building WeCLIP model...")
        return WeCLIPModel()

    def train_step(self):
        print("Training WeCLIP model...")

    def predict(self, image):
        print("Predicting with WeCLIP model...")
        return image
