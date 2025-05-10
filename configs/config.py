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


def load_config(config_path: Path) -> DictConfig:
    """load_config() 加载指定算法配置文件 (覆盖基础配置文件)"""
    base_config = load_base_config()
    user_config = OmegaConf.load(config_path)
    return OmegaConf.merge(base_config, user_config)


def get_config(config_path: Path, args_list: list[str]) -> DictConfig:
    """get_config() 获取配置文件, 并合并命令行参数"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    cli_config = OmegaConf.from_cli(args_list)
    file_config = load_config(config_path)
    return OmegaConf.merge(file_config, cli_config)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = get_config(root / "./excel.yaml", [])
    print(config)
