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

# Torch Library

# My Library
from core import Trainer
from configs.config import get_config
from algorithms import get_algorithm
from algorithms.base import (
    WeaklySupervisedSemanticSegmentationAlgorithm as WSSSAlgorithm,
)

from .args import get_args


def main():
    args, options = get_args()
    config = get_config(args.config, options)

    algorithm: WSSSAlgorithm = get_algorithm(config.model.name)(config)

    Trainer(algorithm=algorithm, config=config).run()


if __name__ == "__main__":
    main()
