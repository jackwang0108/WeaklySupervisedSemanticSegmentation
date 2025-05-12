"""
classes.py 提供了统一接口的类

    @Time    : 2025/05/10
    @Author  : JackWang
    @File    : classes.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from typing import Any, Optional

# Third-Party Library

# Torch Library
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

# My Library


class NullScheduler(scheduler.LRScheduler):
    """
    空的LR调度器，不执行任何操作但提供相同接口

    因为Trainer中会save和load训练状态, 所以对于没有LR Scheduling的算法, 需要屏蔽差异
    """

    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        self._step_count += 1
        self._last_lr: list[float] = self.get_lr()

    def state_dict(self):
        return {"null_scheduler": True}

    def load_state_dict(self, state_dict: dict[str, Any]):
        return
