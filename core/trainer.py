"""
trainer.py 定义通用的训练器类

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : trainer.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library

# Third-Party Library
from omegaconf import DictConfig

# Torch Library

# My Library
from algorithms.base import (
    WeaklySupervisedSemanticSegmentationAlgorithm as WSSSAlgorithm,
)


class Trainer:
    def __init__(self, algorithm: WSSSAlgorithm, config: DictConfig):
        self.config = config
        self.algorithm = algorithm

    def run(self):
        print("Running training pipeline...")
        self.algorithm.train_step()
        print("Training completed.")
