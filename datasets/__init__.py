"""
__init__.py 用于初始化datasets模块

    @Time    : 2025/05/10
    @Author  : JackWang
    @File    : __init__.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from typing import TypeVar, Literal

# Third-Party Library

# Torch Library

# My Library
from .voc import VOC2012WSSSDataset
from .coco import COCO2014WSSSDataset


AvailableDataset = Literal["voc", "coco"]

WSSSDataset = TypeVar(
    "AvailableDataset",
    VOC2012WSSSDataset,
    COCO2014WSSSDataset,
)
