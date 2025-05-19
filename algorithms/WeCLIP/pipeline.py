"""
pipeline.py 定义了WeCLIP算法 (Frozen ClIP: A Strong Backbone for Weakly Supervised Semantic Segmentation) 的训练流程

    @Time    : 2025/05/14
    @Author  : JackWang
    @File    : pipeline.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library

# Third-Party Library
from omegaconf import DictConfig

# Torch Library
import torch
import torch.nn.functional as F

# My Library
from .. import register_algorithm
from ..base import WeaklySupervisedSemanticSegmentationAlgorithm as WSSSAlgorithm

from .model import WeCLIPModel
from .cam_refinement import cam_refinement
from .postprocessing import get_post_processor


def get_label_texts(prompt: str, class_names: list[str]) -> list[str]:
    return [prompt.replace("[CLS]", class_name) for class_name in class_names]


def create_neighbor_mask(h: int, w: int, radius: int) -> torch.Tensor:
    """创建局部邻域掩码 (h*w, h*w)"""
    hw = h * w

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack([x, y], dim=-1)  # (h,w,2)

    coords_flat = coords.view(hw, 2)
    delta = coords_flat.unsqueeze(1) - coords_flat.unsqueeze(0)
    return (delta.abs().max(dim=-1).values <= radius).float()


def pseudo_label_to_affinity_label(
    pseudo_labels: torch.Tensor,
    grid_size: tuple[int, int] | torch.Size,
    radius: int = 8,
    ignore_index: int = 255,
) -> torch.Tensor:
    """将GradCAM生成的伪标签转换为Affinity Label, 对应论文中9式"""
    # downsampled: [B, H, W] -> [B, 1, h, w]
    downsampled: torch.Tensor = F.interpolate(
        pseudo_labels.unsqueeze(dim=1).float(), size=grid_size, mode="nearest"
    )

    # 原文中 Affinity Label 描述的是如果是相同的类别, 则为1, 否则为0
    # 所以直接用转置相等即可. 原文中是转成one-hot, 然后再诸位相乘.
    # 最后相等的是1, 不等的是0, 最终对num_class维度取argmax就可以得到这里的结果
    flattened = downsampled.view(*downsampled.shape[:2], -1, 1)
    flattened = flattened.repeat(1, 1, 1, flattened.shape[-2])
    flattened_t = flattened.transpose(2, 3)
    affinity_label = (flattened == flattened_t).long()

    # 处理无效值和空间领域约束
    attn_mask = create_neighbor_mask(*grid_size, radius).to(downsampled.device)
    invalid_mask = (flattened == ignore_index) | (flattened_t == ignore_index)
    affinity_label[invalid_mask | (attn_mask == 0)] = ignore_index

    return affinity_label.long()


def _upsample(
    segmentation_mask: torch.Tensor, target_size: int | tuple[int, int] | torch.Size
) -> torch.Tensor:
    """
    WeCLIP中模型输出的Segmentation Mask的大小其实是ViT的Patch网格的形状, 与原图并不匹配, 所以需要进行上采样

    F.interpolate 的反向传播是可微分的，梯度会通过插值操作正常传递到低分辨率特征图上。具体表现：

        - 最近邻插值：梯度会直接复制到对应的低分辨率位置（无权重分配）
        - 双线性/双三次插值：梯度会按插值权重分配到周围4个（或16个）低分辨率像素点

    但是双线性插值可能导致梯度"模糊化"，尤其在物体边缘区域, 对高分辨率细节敏感的任务，可以尝试 mode='nearest'

    这里是与原始论文实现对齐
    """

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    return F.interpolate(
        segmentation_mask,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )


def get_affinity_loss(
    affinity_map: torch.Tensor, affinity_label: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_label = (affinity_label == 1).float()
    negative_label = (affinity_label == 0).float()

    # 避免除零
    positive_count: torch.Tensor = positive_label.sum(dim=(1, 2)) + 1
    negative_count: torch.Tensor = negative_label.sum(dim=(1, 2)) + 1

    negative_loss = (negative_label * affinity_map).sum(dim=(1, 2)) / negative_count
    positive_loss = (positive_label * (1 - affinity_map)).sum(
        dim=(1, 2)
    ) / positive_count

    return ((negative_loss + positive_loss).sum() / 2, positive_count, negative_count)


def get_segmentation_loss(
    prediction_mask: torch.Tensor,
    pseudo_gt_mask: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """计算分割损失, 原始论文的实现中是强制让前景和背景的损失贡献相同"""

    background_gt_mask = pseudo_gt_mask.clone()
    background_gt_mask[pseudo_gt_mask != 0] = ignore_index
    background_loss = (
        torch.tensor(0.0, device=prediction_mask.device)
        if torch.all(background_gt_mask == ignore_index)
        else F.cross_entropy(
            prediction_mask, background_gt_mask, ignore_index=ignore_index
        )
    )

    foreground_gt_mask = pseudo_gt_mask.clone()
    foreground_gt_mask[pseudo_gt_mask == 0] = ignore_index
    foreground_loss = (
        torch.tensor(0.0, device=prediction_mask.device)
        if torch.all(foreground_gt_mask == ignore_index)
        else F.cross_entropy(
            prediction_mask, foreground_gt_mask, ignore_index=ignore_index
        )
    )

    return (background_loss + foreground_loss) / 2


@register_algorithm("WeCLIP", True)
class WeCLIP(WSSSAlgorithm):
    def __init__(self, config: DictConfig):
        super().__init__(config=config)

        self.model: WeCLIPModel

        self.cam_to_segmentation_mask = get_post_processor()

        self.info_dict = {}

    def build_model(self) -> WeCLIPModel:

        dataset = self.config.data.dataset_using
        prompt = self.config.algorithm.prompt
        self.foreground_classes = self.config.data.available_datasets[
            dataset
        ].classnames
        self.background_classes = self.config.data.available_datasets[
            dataset
        ].background

        return WeCLIPModel(
            backbone=self.config.algorithm.backbone,
            feature_dim=self.config.algorithm.feature_dim,
            num_classes=self.config.algorithm.num_classes,
            background_labels=get_label_texts(prompt, self.background_classes),
            foreground_labels=get_label_texts(prompt, self.foreground_classes),
        )

    def train_step(
        self, data: list[torch.Tensor], epoch: int, batch: int, num_batches: int
    ):
        # images: [B, 3, H, W]
        (
            original_image,
            augmented_images,
            augmented_segmentation_mask,
            weakly_supervised_label,
        ) = data
        image_shape = augmented_images.shape[-2:]

        # affinity_map: [B, h * w, h * w]
        # attention_maps: [B, 12, h * w, h * w]
        # prediction_mask: [B, num_class, h, w], num_class 含背景
        affinity_map: torch.Tensor
        _, affinity_map, prediction_mask, attention_maps = self.model(augmented_images)

        # prediction_mask: [B, num_class, H, W], num_class 含背景
        prediction_mask = _upsample(prediction_mask, image_shape)

        # cams: [B, num_class - 1, h, w], 不含背景
        cams = self.model.get_gradcams(augmented_images)

        # refined_cams: [B, num_class - 1, h, w], 不含背景
        refined_cams = cam_refinement(
            cams,
            affinity_map,
            attention_maps,
            self.config.algorithm.n0,
            self.config.algorithm.alpha,
        )

        # pseudo_gt_mask: [B, H, W], 含背景
        pseudo_gt_mask = self.cam_to_segmentation_mask(
            refined_cams, augmented_images, image_shape
        )

        # affinity_label: [B, 1, h, w]
        affinity_label = pseudo_label_to_affinity_label(
            pseudo_gt_mask.detach(), cams.shape[-2:]
        ).squeeze(dim=1)

        segmentation_loss = get_segmentation_loss(
            prediction_mask, pseudo_gt_mask.long(), ignore_index=255
        )

        affinity_loss, positive_count, negative_count = get_affinity_loss(
            affinity_map, affinity_label
        )

        loss = segmentation_loss + affinity_loss * self.config.algorithm.lambda_affinity

        return_values = {
            "loss": loss,
            "aff_loss": affinity_loss,
            "seg_loss": segmentation_loss,
        }

        self.writer.add_scalar(
            "train/losses/batch/affinity_loss",
            affinity_loss,
            global_step=epoch * num_batches + batch,
        )
        self.writer.add_scalar(
            "train/losses/batch/segmentation_loss",
            segmentation_loss,
            global_step=epoch * num_batches + batch,
        )
        self.writer.add_scalar(
            "train/losses/batch/total_loss",
            loss,
            global_step=epoch * num_batches + batch,
        )

        self.info_dict |= {
            "epoch": epoch,
            "batch": batch,
            **return_values,
        }
        return return_values

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        # prediction_mask: [B, num_class, H, W], num_class 含背景
        _, _, prediction_mask, _ = self.model(image)
        return _upsample(prediction_mask, self.model.clip.image_resolution).argmax(
            dim=1
        )

    def get_info_dict(self):
        return self.info_dict


if __name__ == "__main__":
    pred = torch.randn(2, 21, 224, 224)
    gt = torch.randint(0, 21, (2, 224, 224))

    print(F.cross_entropy(pred, gt))
