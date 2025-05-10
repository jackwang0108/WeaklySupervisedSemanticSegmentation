"""
transforms.py 定义了数据增强的方法与类

torchvision.transforms 只能处理图像, 无法同步处理Segmentation Mask (例如: RandomCrop)
torchvision.transforms.v2 虽然可以同步处理图像和mask, 但是不支持自定义数据集
所以最终还是需要自己实现可以同步处理图像和mask的数据增强方法

    @Time    : 2025/05/08
    @Author  : JackWang
    @File    : transforms.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import math
import random
from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal
from collections.abc import Callable, Sequence

# Third-Party Library
import numpy as np
import PIL.Image as Image

# Torch Library
import torch
import torchvision.transforms.functional as F

# My Library

__all__ = (
    "Transform",
    "Compose",
    "ToDtype",
    "ToTensor",
    "ToPILImage",
    "Resize",
    "RandomResizedCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "ColorJitter",
    "Normalize",
)


def apply_all(x: Any | Sequence[Any], func: Callable) -> Any | Sequence[Any]:
    return [func(t) for t in x] if isinstance(x, (list, tuple)) else func(x)


class Transform(ABC):
    @abstractmethod
    def __call__(self, x: Any) -> Any:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Compose(Transform):
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(
        self, x: tuple[np.ndarray, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}("
        for t in self.transforms:
            format_string += f"\n    {t},"
        format_string += "\n)"
        return format_string


class ToDtype(Transform):
    def __init__(self, dtype: torch.dtype | np.dtype):
        self.dtype = dtype

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        converter = lambda x: (
            x.astype(self.dtype) if isinstance(x, np.ndarray) else x.to(self.dtype)
        )
        return apply_all(x, converter)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self.dtype})"


class ToTensor(Transform):
    def __init__(self, div: bool = True):
        self.div = div

    def __call__(
        self, x: np.ndarray | tuple[np.ndarray, np.ndarray]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, np.ndarray):
            return F.to_tensor(x) * (1 if self.div else 255)

        image = F.to_tensor(x[0]) * (1 if self.div else 255)
        mask = torch.from_numpy(x[1]).long().unsqueeze(dim=0)
        return (image, mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(div={self.div})"


# fmt: off
PIL_MODES = Literal[
    "1", "CMYK", "F", "HSV", "I", "I;16", "I;16B", "I;16L", "I;16N", "L", 
    "LA", "La", "LAB", "P", "PA", "RGB", "RGBA", "RGBa", "RGBX", "YCbCr"
]
# fmt: on


class ToPILImage(Transform):
    def __init__(self, mode: Optional[PIL_MODES] = None):
        self.mode = mode

    def __call__(
        self, x: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor]
    ) -> Image.Image | Sequence[Image.Image]:
        return apply_all(x, partial(F.to_pil_image, mode=self.mode))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode})"


class Resize(Transform):
    """Resize 图像和mask到指定大小"""

    def __init__(self, size: int | tuple[int, int]):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # image: [3, H, W], mask: [H, W]
        image, mask = x
        image = F.resize(
            image,
            self.size,
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = F.resize(
            mask,
            self.size,
            interpolation=F.InterpolationMode.NEAREST_EXACT,
            antialias=False,
        )
        return image, mask

    def __repr__(self) -> str:
        detail = f"(size={self.size}"
        detail += ", image interpolation=BILINEAR, mask interpolation=NEAREST_EXACT"
        detail += ", image antialias=True, mask antialias=False)"
        return f"{self.__class__.__name__}{detail}"


class RandomResizedCrop(Transform):
    def __init__(
        self,
        size: int | tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        antialias: Optional[bool] = True,
    ):
        """
        RandomResizedCrop 通过随机裁剪和缩放来增强图像, 使得模型更具鲁棒性

        scale: 裁剪区域的比例范围
            - 例如 (0.08, 1.0) 表示裁剪区域的面积占原图像面积的 8% 到 100%
            - 让模型学习不同尺寸的物体（如近景的大物体 vs 远景的小物体）
            - 若固定比例（如始终裁剪50%），模型会过度依赖特定尺寸特征。

        ratio: 裁剪区域的宽高比范围
            - 例如 (3 / 4, 4 / 3) 表示裁剪区域的宽高比在 3:4 到 4:3 之间
            - 现实物体的宽高比各异（如人像 vs 风景照片）。
            - 固定比例（如1:1正方形）会导致模型对形状变化敏感。
            - 限制宽高比范围可避免生成极端扁长或窄高的裁剪区域（如10:1）
                这类区域在resize时会严重扭曲内容。

        """
        self.scale = scale
        self.ratio = ratio
        self.antialias = antialias
        self.size = (size, size) if isinstance(size, int) else size

    def make_params(self, image: torch.Tensor) -> dict[str, int]:
        height, width = image.shape[-2:]

        # 宽 = 面积 * 宽高比, 高 = 面积 / 宽高比
        area = height * width
        log_ratio = [math.log(r) for r in self.ratio]

        # try to sample a random area 10 times
        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            # if the area meets all requirements
            # get the upper left corner, and then, break
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h + 1)
                j = random.randint(0, width - w + 1)
                break
        else:
            # fallback to center crop
            initial_ratio = width / height
            if initial_ratio < min(self.ratio):
                w, h = width, int(round(width / min(self.ratio)))
            elif initial_ratio > max(self.ratio):
                h, w = height, int(round(height * max(self.ratio)))
            else:
                w, h = width, height
            i, j = (height - h) // 2, (width - w) // 2

        return dict(top=i, left=j, height=h, width=w)

    def __call__(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # image: [3, H, W], mask: [H, W]
        image, mask = x
        params = self.make_params(image)

        image = F.resized_crop(
            image,
            **params,
            size=self.size,
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=self.antialias,
        )

        mask = F.resized_crop(
            mask,
            **params,
            # 严格最近邻插值, 避免出现0.5这样的值
            interpolation=F.InterpolationMode.NEAREST_EXACT,
            size=self.size,
            antialias=False,  # mask不需要抗锯齿
        )
        return image, mask

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += (
            ", image interpolation=BILINEAR, mask interpolation=NEAREST_EXACT"
        )
        format_string += ", image antialias=True, mask antialias=False)"
        return format_string


class RandomHorizontalFlip(Transform):
    """随机水平翻转图像和mask"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return [F.hflip(t) for t in x] if random.random() < self.p else x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip(Transform):
    """随机垂直翻转图像和mask"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return [F.vflip(t) for t in x] if random.random() < self.p else x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotation(Transform):
    """随机旋转图像和mask"""

    def __init__(
        self,
        degrees: float | tuple[float, float],
        expand: bool = False,
        center: Optional[list[float]] = None,
    ):
        self.expand = expand
        self.center = center
        self.degrees = (
            degrees
            if isinstance(degrees, (list, tuple))
            else (-abs(degrees), abs(degrees))
        )

    def make_params(self) -> dict[str, int]:
        angle = random.uniform(*self.degrees)
        return dict(angle=angle)

    def __call__(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # image: [3, H, W], mask: [H, W]
        image, mask = x

        params = self.make_params()

        image = F.rotate(
            image,
            **params,
            interpolation=F.InterpolationMode.NEAREST,
            expand=self.expand,
            center=self.center,
            fill=[0, 0, 0],
        )

        mask = F.rotate(
            mask,
            **params,
            interpolation=F.InterpolationMode.NEAREST,
            expand=self.expand,
            center=self.center,
            fill=255,
        )
        return image, mask

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}(degrees={self.degrees}"
        format_string += ", image interpolation=NEAREST, mask interpolation=NEAREST"
        format_string += f", expand={self.expand}"
        if self.center is not None:
            format_string += f", center={self.center}"
        format_string += ", image fill=(0, 0, 0), mask fill=255"
        format_string += ")"
        return format_string


class ColorJitter(Transform):
    """随机改变图像的亮度、对比度、饱和度和色调"""

    from torchvision.transforms import ColorJitter as _ColorJitter

    def __init__(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
    ):
        self.color_jitter = self._ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.color_jitter(x[0]), x[1]

    def __repr__(self):
        return self.color_jitter.__repr__()


class Normalize(Transform):
    """
    对图像进行归一化处理

    ToTensor() 调用 torchvision.transforms.functional.to_tensor() 将图像转换为Tensor时, 只是单纯的将像素值从0~255转换为0~1

    因此, 后序需要使用Normalize() 将0~1的像素分布转换为均值为0, 方差为1的分布
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return F.normalize(x[0], self.mean, self.std), x[1]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


if __name__ == "__main__":
    from .voc import VOC2012WSSSDataset
    from .coco import COCO2014WSSSDataset

    transforms = Compose(
        [
            ToTensor(),
            RandomResizedCrop(size=256),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=30, expand=True),
        ]
    )

    rgb_visualizer = ToPILImage("RGB")
    mask_visualizer = Compose(
        [
            ToDtype(torch.uint8),
            ToPILImage(mode="L"),
        ]
    )

    voc_ds = VOC2012WSSSDataset(split="train_aug", transform=transforms)
    coco_ds = COCO2014WSSSDataset(split="train", transform=transforms)

    print(len(voc_ds), len(coco_ds))

    voc_image, voc_mask, _ = voc_ds[0]
    coco_image, coco_mask, _ = coco_ds[0]

    rgb_visualizer(voc_image).show()
    mask_visualizer(voc_mask).show()

    rgb_visualizer(coco_image).show()
    mask_visualizer(coco_mask).show()
