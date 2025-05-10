"""
helper.py 定义了辅助函数和工具类

    @Time    : 2025/05/10
    @Author  : JackWang
    @File    : helper.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import os
import random
from typing import Optional

# Third-Party Library
import numpy as np

# Torch Library
import torch

# My Library


def set_random_seed(seed: int, verbose: Optional[bool] = False) -> None:
    """为了确保可重复性，固定所有随机种子"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 牺牲cuDNN的性能来确保可重复性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 分布式训练时，确保所有进程使用相同的随机种子
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.all_reduce(torch.tensor(seed))

    if verbose:
        print(f"Set random seed to {seed}")
