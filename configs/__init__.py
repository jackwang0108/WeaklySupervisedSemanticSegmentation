"""
__init__.py 对configs模块进行初始化

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
from .config import (
    build_config,
    load_algo_config,
    load_base_config,
)


__all__ = [
    "build_config",
    "load_algo_config",
    "load_base_config",
]


install(word_wrap=True)
