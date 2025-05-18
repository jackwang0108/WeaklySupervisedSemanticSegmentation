"""
coco.py 定义了COCO数据集加载函数和类

@Time    : 2025/05/08
@Author  : JackWang
@File    : coco.py
@IDE     : VsCode
@Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import os
import sys
from pathlib import Path
from collections.abc import Callable
from contextlib import contextmanager
from typing import Optional, Literal, Any

# Third-Party Library
import numpy as np
import PIL.Image as Image
from pycocotools.coco import COCO

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


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout = original_stdout


class COCO2014Dataset(Dataset):
    """COCO2014Dataset reads the original image and corresponding annotations of the COCO 2014 dataset."""

    def __init__(
        self,
        root: Optional[Path] = None,
        split: Optional[Literal["train", "val"]] = "train",
    ):
        super().__init__()

        self.split = split
        self.root = load_data_dir("coco") if root is None else root
        self.image_name_list = load_image_list("coco", split)

        self.json_path = self.root / f"annotations/instances_{self.split}2014.json"
        with suppress_stdout():
            self.coco = COCO(self.json_path)

        # COCO 的 category_id 是1-90的不连续数字, 一共有80个, 需要重新映射回0-79的连续数字
        self.cat_id_to_label = {
            cat_id: i + 1 for i, cat_id in enumerate(self.coco.getCatIds())
        }

        self.label_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_label.items()}

    def getLabelInfo(
        self, label: Optional[int] = None, cat: Optional[int] = None
    ) -> dict[str, str | int]:
        assert (
            label is not None or cat is not None
        ), "label and cat cannot be None at the same time"
        return self.coco.loadCats(
            self.label_to_cat_id[label] if label is not None else cat
        )[0] | {"label": label}

    def __len__(self) -> int:
        return len(self.image_name_list)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray, list[dict[str, Any]]]:
        image_name = self.image_name_list[index]
        image_path = self.root / f"{self.split}2014/{image_name}.jpg"
        image_id = int(image_name.split("_")[-1])
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        image = Image.open(image_path).convert("RGB")

        return (image_name, np.array(image), annotations)


class COCO2014WSSSDataset(COCO2014Dataset):
    """COCO2014WSSSDataset is the weakly supervised semantic segmentation dataset of COCO 2014."""

    def __init__(
        self,
        root: Optional[Path] = None,
        split: Optional[Literal["train", "val"]] = "train",
        transform: Optional[Callable] = None,
    ):
        super().__init__(root, split)

        self.transform = (
            transform if transform is not None else Compose([ToTensor(), Resize(256)])
        )

        self.original_transform = Compose(
            [i for i in self.transform.transforms if not i._change_image_color]
        )

    def __repr__(self):
        format_string = f"{self.__class__.__name__}("
        format_string += f"\n    root={self.root},"
        format_string += f"\n    split={self.split},"
        format_string += f"\n    length={len(self)},"
        format_string += f"\n    transform="

        for line in repr(self.transform).splitlines():
            format_string += f"\n        {line}"

        format_string += "\n)"
        return format_string

    def get_weakly_supervision_label(self, semantic_mask: torch.Tensor) -> torch.Tensor:
        """获得弱监督标签, 这里的弱监督标签是图像级的物体类别one-hot向量"""
        image_classes = semantic_mask.unique()
        image_classes = image_classes[(image_classes != 255)]

        one_hot = torch.zeros(81, dtype=torch.long)
        if len(image_classes) > 0:
            one_hot[image_classes] = 1
        return one_hot

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note:
            语义分割的label, 即segmentation mask, 是一个二维的Tensor. 其中的0表示背景类, 物体类的label是从1开始的
        """
        _, image, annotations = super().__getitem__(index)

        semantic_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for annotation in annotations:
            label = self.cat_id_to_label[annotation["category_id"]]
            instance_mask = self.coco.annToMask(annotation)
            semantic_mask = np.where(instance_mask, label, semantic_mask)

        original_image, _ = self.original_transform((image, semantic_mask))
        augmented_image, augmented_semantic_mask = self.transform(
            (image, semantic_mask)
        )

        return (
            original_image,
            augmented_image,
            augmented_semantic_mask,
            self.get_weakly_supervision_label(augmented_semantic_mask),
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .transforms import *

    t = Compose(
        [
            ToTensor(),
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ColorJitter(0.2, 0.2, 0.2, 0.1),
            Normalize(
                [0.4573, 0.4373, 0.4045],
                [0.2675, 0.2643, 0.2780],
            ),
        ]
    )

    ds = COCO2014WSSSDataset(split="val", transform=t)

    loader = DataLoader(ds, 32, True, num_workers=1)
    original, images, labels, weak_labels = next(iter(loader))

    rgb_visualizer = ToPILImage("RGB")
    mask_visualizer = Compose(
        [
            ToDtype(torch.uint8),
            ToPILImage("L"),
        ]
    )

    rgb_visualizer(original[0]).show()
    rgb_visualizer(images[0]).show()
