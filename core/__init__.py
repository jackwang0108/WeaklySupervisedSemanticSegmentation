"""
__init__.py 对core模块进行初始化

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : __init__.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library

# Third-Party Library
from rich.traceback import install

# Torch Library

# My Library
from .trainer import Trainer

__all__ = [
    "Trainer",
]

install(word_wrap=True)
