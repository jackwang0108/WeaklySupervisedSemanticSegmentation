"""
__init__.py 对algorithms模块进行初始化

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : __init__.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import pkgutil
import importlib
from pathlib import Path
from warnings import warn
from typing import Literal, Type

# Third-Party Library
from rich.traceback import install

# Torch Library

# My Library
from .base import WeaklySupervisedSemanticSegmentationAlgorithm


install(word_wrap=True)

# 注册算法的字典
ALGORITHMS_REGISTRY: dict[str, Type[WeaklySupervisedSemanticSegmentationAlgorithm]] = {}

# 忽略的模块列表
IGNORE_MODULES = ["pytorch_grad_cam"]


def register_algorithm(algorithm_name: str, throw_warning: bool = False):
    """注册算法的装饰器"""

    def decorator(cls):
        if algorithm_name in ALGORITHMS_REGISTRY:
            if throw_warning:
                warn(f"Warning: Algorithm {algorithm_name} is already registered.")
            else:
                raise ValueError(f"Algorithm {algorithm_name} already registered.")
        ALGORITHMS_REGISTRY[algorithm_name] = cls
        cls.algorithm_name = algorithm_name
        return cls

    return decorator


def _auto_discover_algorithms():
    """自动发现所有算法"""
    package_path = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        if (
            module_name.startswith("_")
            or module_name == "base"
            or module_name in IGNORE_MODULES
        ):
            continue

        try:
            _ = importlib.import_module(f"algorithms.{module_name}.pipeline")
        except ImportError as e:
            print(f"Warning: Failed to import {module_name}: {e}")


_auto_discover_algorithms()


Algorithms = Literal[
    "EXAMPLE",
    "WeCLIP",
    "ExCEL",
]


def get_algorithm(algorithm_name: Algorithms):
    """获取算法的工厂函数"""
    if algorithm_name not in ALGORITHMS_REGISTRY:
        available = list(ALGORITHMS_REGISTRY.keys())
        raise ValueError(
            f"Algorithm {algorithm_name} not found. Available algorithms: {available}"
        )
    return ALGORITHMS_REGISTRY[algorithm_name]
