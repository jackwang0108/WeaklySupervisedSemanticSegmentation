"""
statistics.py 统计数据集的信息

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : statistics.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from pathlib import Path
from collections.abc import Callable

# Third-Party Library

# Torch Library
import torch

# My Library
from .common import load_data_dir, load_image_list
from .voc import VOC2012WSSSDataset
from .coco import COCO2014WSSSDataset
from .transforms import Compose, Resize, ToTensor


def print_sep(callable: Callable):
    def wrapper():
        print(f"{callable.__name__:-^80}")
        callable()
        print("-" * 80)

    return wrapper


@print_sep
def check_voc():
    root = load_data_dir("voc")

    val_list = load_image_list("voc", "val")
    train_list = load_image_list("voc", "train_aug")
    jpeg_list = [i.stem for i in root.glob("JPEGImages/*.jpg")]

    print(f"{len(train_list)=}, {len(val_list)=}, {len(jpeg_list)=}")

    print(f"All val images in JPEGImages? {all(i in jpeg_list for i in val_list)}")
    print(f"All train images in JPEGImages? {all(i in jpeg_list for i in train_list)}")


@print_sep
def check_coco():
    root = load_data_dir("coco")

    val_list = load_image_list("coco", "val")
    train_list = load_image_list("coco", "train")
    val_image_list = [i.stem for i in root.glob("val2014/*.jpg")]
    train_image_list = [i.stem for i in root.glob("train2014/*.jpg")]

    print(f"{len(train_list)=}, {len(val_list)=}")
    print(f"{len(train_image_list)=}, {len(val_image_list)=}")
    print(f"All val images in val2014? {all(i in val_image_list for i in val_list)}")
    print(
        f"All train images in train2014? {all(i in train_image_list for i in train_list)}"
    )


@print_sep
def get_mean_std():
    # 计算数据集均值

    def calculate_mean_std(ds: VOC2012WSSSDataset | COCO2014WSSSDataset):

        count = 0  # 总像素数
        psum = torch.zeros(3)  # 各通道像素和
        psum_sq = torch.zeros(3)  # 各通道像素平方和
        images = []
        for i in range(len(ds)):
            # image: [3, H, W]
            image, _, _ = ds[i]
            psum += image.sum(dim=(1, 2))
            psum_sq += (image**2).sum(dim=(1, 2))
            count += image.shape[1] * image.shape[2]

            print(f"Processed {i}/{len(ds)} images")

        mean = psum / count
        std = (psum_sq / count - mean**2).sqrt()

        return mean, std

    transform = Compose(
        [
            ToTensor(),
            Resize(size=256),
        ]
    )

    mean, std = calculate_mean_std(
        VOC2012WSSSDataset(split="train_aug", transform=transform)
    )
    print(f"VOC2012WSSSDataset mean: {mean}, std: {std}")

    mean, std = calculate_mean_std(
        COCO2014WSSSDataset(split="train", transform=transform)
    )
    print(f"COCO2014WSSSDataset mean: {mean}, std: {std}")


def main():

    check_voc()
    check_coco()
    get_mean_std()


if __name__ == "__main__":
    main()
