"""
voc.py 定义了VOC数据集加载函数和类

    @Time    : 2025/05/07
    @Author  : JackWang
    @File    : voc.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import warnings
from pathlib import Path
from typing import Optional, Literal
from collections.abc import Callable

# Third-Party Library
import numpy as np
import PIL.Image as Image

# Torch Library
import torch
from torch.utils.data import Dataset

# My Library
from .common import (
    load_data_dir,
    load_image_list,
    load_label_dict,
)
from .transforms import (
    Resize,
    Compose,
    ToTensor,
)


class VOC2012Dataset(Dataset):
    """VOC2012Dataset reads the original image and label of the PASCAL VOC 2012 dataset."""

    def __init__(
        self,
        root: Optional[Path] = None,
        split: Optional[Literal["train", "val", "train_aug"]] = "train_aug",
    ):
        super().__init__()

        if split == "train":
            warnings.warn(
                "Are your sure you want to use the original train set instead of the SBD-expanded train set? "
                "Reads README.md for more details."
            )

        self.split = split
        self.root = load_data_dir("voc") if root is None else root
        self.image_name_list = load_image_list("voc", split)

    def __len__(self) -> int:
        return len(self.image_name_list)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray, np.ndarray]:
        image_name = self.image_name_list[index]
        image_path = self.root / "JPEGImages" / f"{image_name}.jpg"
        label_path = self.root / "SegmentationClassAug" / f"{image_name}.png"

        image = np.array(Image.open(image_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("L"))

        return image_name, image, label


class VOC2012WSSSDataset(VOC2012Dataset):
    """VOC2012WSSSDataset is the weakly supervised semantic segmentation dataset of PASCAL VOC 2012."""

    def __init__(
        self,
        root: Optional[Path] = None,
        split: Optional[Literal["train", "val", "train_aug"]] = "train_aug",
        transform: Optional[Callable] = None,
    ):
        super().__init__(root, split)

        self.transform = (
            transform if transform is not None else Compose([ToTensor(), Resize(256)])
        )

    def get_weakly_supervision_label(self, label: torch.Tensor) -> torch.Tensor:
        """获得弱监督标签, 这里的弱监督标签是图像级的物体类别one-hot向量"""
        image_classes = label.unique()
        image_classes = image_classes[(image_classes != 0) & (image_classes != 255)]

        one_hot = torch.zeros(20, dtype=torch.long)
        if len(image_classes) > 0:
            one_hot[image_classes - 1] = 1
        return one_hot

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note:
            语义分割的label, 即segmentation mask, 是一个二维的Tensor. 其中的0表示背景类, 物体类的label是从1开始的

            但是弱监督标签, 即图像级的物体类别one-hot向量, 是label转的一维的Tensor. 物体类的label直接从0开始
        """
        name, image, label = super().__getitem__(index)
        image, label = self.transform((image, label))

        return image, label, self.get_weakly_supervision_label(label)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = VOC2012WSSSDataset(split="val")

    loader = DataLoader(ds, 32, True, num_workers=1)
    images, labels, weak_labels = next(iter(loader))
    print(images.shape, labels.shape, weak_labels.shape)
