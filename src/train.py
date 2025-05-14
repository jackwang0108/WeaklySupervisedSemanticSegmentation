"""
train.py 用于训练指定的模型和算法

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : train.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from pathlib import Path

# Third-Party Library
from omegaconf import DictConfig

# Torch Library

# My Library
from core.trainer import Trainer
from configs.config import build_config
from algorithms import get_algorithm
from algorithms.base import (
    WeaklySupervisedSemanticSegmentationAlgorithm as WSSSAlgorithm,
)

from .args import get_args
from .helper import set_random_seed


def get_config() -> DictConfig:
    args, options = get_args()
    config = build_config(args.config, options)
    return config


def main():
    config = get_config()
    set_random_seed(config.train.seed)
    algorithm: WSSSAlgorithm = get_algorithm(config.algorithm.name)(config)

    Trainer(algorithm=algorithm, config=config).train()


if __name__ == "__main__":
    main()
