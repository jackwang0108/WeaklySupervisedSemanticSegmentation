"""
common.py 定义了数据集的公共函数和常量

    @Time    : 2025/05/07
    @Author  : JackWang
    @File    : common.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from pathlib import Path
from typing import Literal, TypeAlias

# Third-Party Library
import numpy as np

# Torch Library
import torch

# My Library

DATASETS_DIR = Path(__file__).resolve().parent
DATASET_CONFIG_DIR = {
    "voc": DATASETS_DIR / "voc",
    "coco": DATASETS_DIR / "coco",
}

DatasetName: TypeAlias = Literal["voc", "coco"]


def load_data_dir(which_dataset: DatasetName) -> Path:
    """
    load_data_dir 读取默认的数据集路径

    Args:
        which_dataset (DatasetName): 数据集名称

    Raises:
        FileNotFoundError: 在默认路径下找不到数据集

    Returns:
        Path: 数据集路径
    """
    try:
        with (DATASET_CONFIG_DIR[which_dataset] / f"{which_dataset}.txt").open(
            "r"
        ) as f:
            return [Path(line.strip()) for line in f.readlines()][0]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Dataset {which_dataset.upper()} path not found in default dir. Please run scripts/download_dataset.sh to download the datasets or set dataset dir manually."
        ) from e


def load_image_list(
    which_dataset: DatasetName,
    which_file: Literal["test", "train", "trainval", "train_aug", "val"],
) -> list[str]:
    """
    load_image_list 读取预先划分好的数据集split的文件列表

    Args:
        which_dataset (DatasetName): 数据集名称
        which_file (Literal[&quot;test&quot;, &quot;train&quot;, &quot;trainval&quot;, &quot;train_aug&quot;, &quot;val&quot;]): split名称

    Returns:
        list[str]: 该split中图像名称的列表
    """
    return np.loadtxt(
        DATASET_CONFIG_DIR[which_dataset] / f"{which_file}.txt", dtype=str
    ).tolist()


def load_label_dict(which_dataset: DatasetName) -> dict[str, np.ndarray[int]]:
    """
    load_label_dict 读取数据集的标签

    返回一个字典: {image name: one-hot vector}
    因为是给定图像标签的弱监督语义分割, 所以每个one-hot向量有多个1, 表示该图像中所有的物体类别

    Args:
        which_dataset (DatasetName): 数据及名称

    Returns:
        dict[str, np.ndarray[int]]: 数据集的标签
    """
    loaded: np.ndarray = np.load(
        DATASET_CONFIG_DIR[which_dataset] / "class_labels_onehot.npy",
        allow_pickle=True,
    )
    return loaded.item()


def to_onehot(label: torch.Tensor, num_classes: int) -> torch.Tensor:

    pass


if __name__ == "__main__":
    # Example usage
    image_name_list = load_image_list("voc", "train_aug")
    print(image_name_list[:5], len(image_name_list))

    class_label_list = load_label_dict("voc")
    print(class_label_list[:5], len(class_label_list))
