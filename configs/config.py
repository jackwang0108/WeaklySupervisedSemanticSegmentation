"""
config.py 定义了配置文件的加载和解析

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : config.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from pathlib import Path

# Third-Party Library
from omegaconf import OmegaConf, DictConfig

# Torch Library

# My Library


def load_base_config() -> DictConfig:
    """load_base_config() 加载通用算法的基础配置文件"""
    return OmegaConf.load(Path(__file__).resolve().parent / "base.yaml")


def load_algo_config(config_path: Path) -> DictConfig:
    """load_algo_config() 加载指定算法配置文件 (覆盖基础配置文件)"""
    base_config = load_base_config()
    algo_config = OmegaConf.load(config_path)
    return OmegaConf.merge(base_config, algo_config)


def build_config(config_path: Path, cmd_args: list[str]) -> DictConfig:
    """build_config() 从配置文件和命令行参数中构建得到训练配置"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    cli_config = OmegaConf.from_cli(cmd_args)
    file_config = load_algo_config(config_path)
    return OmegaConf.merge(file_config, cli_config)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = build_config(root / "./excel.yaml", [])
    print(config)
