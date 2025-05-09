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

# My Library
from .common import load_data_dir, load_image_list


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


def main():

    check_voc()
    check_coco()


if __name__ == "__main__":
    main()
