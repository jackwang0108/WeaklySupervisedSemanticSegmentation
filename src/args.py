"""
args.py 用于获得命令行的参数

    @Time    : 2025/05/10
    @Author  : JackWang
    @File    : args.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import argparse
from pathlib import Path

# Third-Party Library

# Torch Library

# My Library


def get_args() -> tuple[argparse.Namespace, list[str]]:
    """get_args() 用于获得命令行的参数"""
    parser = argparse.ArgumentParser(
        description="Weakly Supervised Semantic Segmentation (WSSS)"
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the config file (yaml) to train the model",
    )

    args, options = parser.parse_known_args()

    for option in options:
        assert (
            "=" in option
        ), f"Invalid option: {option}, should be like field1.field2=value"
        field, value = option.split("=")
        assert field != "", f"Invalid option: {option}, no field provided"
        assert value != "", f"Invalid option: {option}, no value provided"

    return args, options
