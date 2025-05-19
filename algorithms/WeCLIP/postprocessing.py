"""
postprocessing.py 对GradCAM得到的激活图像进行后处理, 从而得到伪标签

    @Time    : 2025/05/17
    @Author  : JackWang
    @File    : postprocessing.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from collections.abc import Callable

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
import torch.nn.functional as F

# My Library


def get_kernel() -> torch.Tensor:

    weight = torch.zeros(8, 1, 3, 3)
    weight[0, 0, 0, 0] = 1
    weight[1, 0, 0, 1] = 1
    weight[2, 0, 0, 2] = 1

    weight[3, 0, 1, 0] = 1
    weight[4, 0, 1, 2] = 1

    weight[5, 0, 2, 0] = 1
    weight[6, 0, 2, 1] = 1
    weight[7, 0, 2, 2] = 1

    return weight


class PixelAdaptiveRefinement(nn.Module):

    def __init__(self, dilations, num_iter: int):
        super().__init__()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

        self.num_iter = num_iter
        self.dilations = dilations

        self.pos = self.get_pos()

        kernel = get_kernel()
        self.kernel: torch.Tensor
        self.register_buffer("kernel", kernel)

    def get_dilated_neighbors(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode="replicate", value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        return torch.cat([ker * d for d in self.dilations], dim=2)

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:

        # masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        images = F.interpolate(
            images, size=masks.size()[-2:], mode="bilinear", align_corners=True
        )

        b, c, h, w = images.shape
        _imgs = self.get_dilated_neighbors(images)
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = images.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _pos_rep = _pos.repeat(b, 1, 1, h, w)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)

        aff = -((_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2)
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -((_pos_rep / (_pos_std + 1e-8) / self.w1) ** 2)
        # pos_aff = pos_aff.mean(dim=1, keepdim=True)

        aff = F.softmax(aff, dim=2) + self.w2 * F.softmax(pos_aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks


def get_post_processor() -> (
    Callable[[torch.Tensor, torch.Tensor, tuple[int, int] | torch.Size], torch.Tensor]
):
    first_run = True
    processor = PixelAdaptiveRefinement(num_iter=20, dilations=[1, 2, 4, 8, 12, 24])

    def post_processor(
        cam: torch.Tensor,
        image: torch.Tensor,
        target_size: int | tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        """
        GradCAM结果只是激活图, 为了能够训练, 因此需要将GradCAM激活图转为伪标签
        """

        # 映射到0~1, [B, num_class - 1, h, w]
        cam = (cam - torch.amin(cam, dim=(-2, -1), keepdim=True)) / (
            1e-7 + torch.amax(cam, dim=(-2, -1), keepdim=True)
        )

        # 上采样到原图大小, [B, num_class - 1, H, W]
        upscaled_cam: torch.Tensor
        upscaled_cam = F.interpolate(
            cam,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # upscaled_cam 不含背景类, 因此需要添加背景类到头部, [B, num_class, H, W]
        background_score = 1 - upscaled_cam.max(dim=1, keepdim=True).values
        upscaled_cam = torch.cat([background_score, upscaled_cam], dim=1)

        nonlocal first_run, processor
        if first_run:
            processor = processor.to(cam.device)
            first_run = False

        return processor(image, upscaled_cam).argmax(dim=1)

    return post_processor
