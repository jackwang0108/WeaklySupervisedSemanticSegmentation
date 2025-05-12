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
from typing import Any, Optional
from collections.abc import Set, Sequence, Mapping

# Third-Party Library
import numpy as np
from omegaconf import ListConfig, DictConfig

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

# My Library
from .classes import NullScheduler

from datasets import (
    WSSSDataset,
    AvailableDataset,
)
from datasets.transforms import *
from datasets.voc import VOC2012WSSSDataset
from datasets.coco import COCO2014WSSSDataset


def get_freer_gpu() -> int:
    """获取空闲的GPU设备ID"""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU | grep Used >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    device_id = np.argmin(memory_available)
    os.system("rm tmp")
    return device_id


def get_device() -> torch.device:
    """获取当前可用的设备"""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{get_freer_gpu()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_dtype(dtype: str) -> torch.dtype:
    """获取指定的torch数据类型"""
    if dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def to(
    obj: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Any:
    """将对象转换到指定的设备和数据类型"""
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device

    if isinstance(obj, (str, int, float, bool, DictConfig, ListConfig)):
        return obj
    if isinstance(obj, (nn.Module, torch.Tensor)):
        return obj.to(**kwargs)
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device=device, dtype=dtype)

    if isinstance(obj, (Set, Sequence)):
        return type(obj)(to(item, dtype=dtype, device=device) for item in obj)
    if isinstance(obj, dict):
        return type(obj)(
            (key, to(value, dtype=dtype, device=device)) for key, value in obj.items()
        )

    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in vars(obj).items():
            try:
                setattr(obj, attr_name, to(attr_value, dtype=dtype, device=device))
            except (AttributeError, TypeError):
                continue
        return obj

    return obj


def get_transforms(transform_configs: ListConfig) -> Transform:
    transforms = []
    for tc in transform_configs:
        transform_type = globals()[tc.name]
        transforms.append(
            transform_type(
                **{
                    k: (tuple(v) if isinstance(v, ListConfig) else v)
                    for k, v in tc.params.items()
                }
            )
            if "params" in tc
            else transform_type()
        )

    return Compose(transforms)


def get_dataset(
    datasets: AvailableDataset, transforms: Optional[tuple[Transform, Transform]] = None
) -> tuple[WSSSDataset, WSSSDataset]:

    val_transform, train_transform = (None, None) if transforms is None else transforms

    if datasets == "voc":
        val_dataset = VOC2012WSSSDataset(split="val", transform=val_transform)
        train_dataset = VOC2012WSSSDataset(split="train_aug", transform=train_transform)
    elif datasets == "coco":
        val_dataset = COCO2014WSSSDataset(split="val", transform=val_transform)
        train_dataset = COCO2014WSSSDataset(split="train", transform=train_transform)
    else:
        raise ValueError(f"Unsupported dataset: {datasets}")

    return val_dataset, train_dataset


def get_optimizer(
    parameters: list[nn.Parameter], optimizer_config: DictConfig
) -> optim.Optimizer:
    return getattr(optim, optimizer_config.name)(parameters, **optimizer_config.params)


def get_scheduler(
    optimizer: optim.Optimizer, scheduler_config: Optional[DictConfig] = None
) -> scheduler.LRScheduler:
    if scheduler_config is None or scheduler_config.get("name") is None:
        return NullScheduler(optimizer)

    return getattr(optim.lr_scheduler, scheduler_config.name)(
        optimizer, **scheduler_config.params
    )
