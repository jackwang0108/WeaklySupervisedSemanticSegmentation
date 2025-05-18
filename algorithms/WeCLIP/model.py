"""
model.py 定义了WeCLIP算法 (Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation) 的网络架构

@Time    : 2025/05/14
@Author  : JackWang
@File    : model.py
@IDE     : VsCode
@Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from copy import deepcopy
from typing import Optional
from collections import OrderedDict
from collections.abc import Callable

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn

# My Library
from . import clip
from ..pytorch_grad_cam import GradCAM
from .clip.model import CLIP, VisionTransformer
from ..pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class MLP(nn.Module):
    """
    MLP 是使用torch.einsum实现的MLP, 避免来回reshape/view/permute, 使得代码更简洁.

    对应论文Figure 2中蓝色Decoder中几个黄色的`M`
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.projection1 = nn.Linear(in_features, out_features)
        self.projection2 = nn.Linear(out_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # [B, h, w, in_features] ->  [B, h, w, out_features]
        feature_map = (
            torch.einsum("bhwi,oi->bhwo", feature_map, self.projection1.weight)
            + self.projection1.bias
        )
        feature_map = self.activation(feature_map)
        # [B, h, w, out_features] ->  [B, h, w, out_features]
        feature_map = (
            torch.einsum("bhwi,oi->bhwo", feature_map, self.projection2.weight)
            + self.projection2.bias
        )
        return feature_map


class FeatureFusion(nn.Module):
    """
    FeatureFusion 对CLIP的Image Encoder得到的同一图像的不同特征进行融合.

    该模块对应论文Figure 2中蓝色Decoder的左侧几个`M`和`C`, 对应Equation 1和2s
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 768,
        num_features: int = 12,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_features = num_features

        self.mlps = nn.ModuleList([MLP(in_dim, out_dim) for _ in range(num_features)])

        self.conv = nn.Conv2d(
            in_channels=num_features * out_dim,
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # [B, num_layer, h, w, in_features] ->  [B, num_layer, h, w, out_features]
        feature_map = torch.stack(
            [mlp(lfm) for mlp, lfm in zip(self.mlps, feature_map.transpose(0, 1))],
            dim=1,
        )
        # 通过一维卷积进行特征融合
        # [B, num_layer, h, w, out_features] ->  [B, num_layer * out_features, h, w]
        feature_map = feature_map.permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)

        # [B, num_layer * out_features, h, w] ->  [B, h, w, out_features]
        feature_map = self.conv(feature_map).permute(0, 2, 3, 1)

        return self.dropout(feature_map)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, feature_dim: int, num_heads: int, attn_mask: torch.Tensor = None
    ):
        """
        CLIP的Self-Attention实现

        Args:
            d_model (int): 输入token序列中每个token的维度, 文本是512, 图像是768
            n_head (int): MultiheadAttention的头数, 文本默认是8, 图像默认是12
            attn_mask (torch.Tensor, optional): _description_. Defaults to None.
        """
        super().__init__()

        self.num_heads = num_heads
        self.feature_dim = feature_dim

        self.attn = nn.MultiheadAttention(feature_dim, num_heads)
        self.ln_1 = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(feature_dim, feature_dim * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(feature_dim * 4, feature_dim)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(feature_dim)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FeatureDecoder(nn.Module):
    """
    FeatureDecoder 对CLIP的Image Encoder得到的同一图像的不同特征融合之后再解码, 得到最终的输出.

    该模块对应论文Figure 2中蓝色Decoder部分
    """

    def __init__(
        self,
        in_dim: int = 768,
        mid_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 3,
        num_features: int = 12,
        num_classes: int = 21,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_features = num_features

        self.feature_fusion = FeatureFusion(
            in_dim=in_dim, out_dim=mid_dim, num_features=num_features
        )

        self.transformer_blocks = nn.Sequential(
            *[ResidualAttentionBlock(mid_dim, num_heads) for _ in range(num_layers)]
        )

        self.linear_projection = nn.Conv2d(
            mid_dim, num_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        fused_feature_map: torch.Tensor

        # [B, num_layer, h, w, in_dim] ->  [B, h, w, mid_dim]
        fused_feature_map = self.feature_fusion(feature_map)

        # [B, h, w, mid_dim] ->  [h * w, B, mid_dim]
        b, h, w = fused_feature_map.shape[:3]
        feature_map = (
            fused_feature_map.clone().flatten(start_dim=1, end_dim=2).permute(1, 0, 2)
        )

        # [h * w, B, mid_dim] ->  [h * w, B, mid_dim]
        feature_map = self.transformer_blocks(feature_map)

        # [h * w, B, mid_dim] ->  [B, mid_dim, h, w]
        feature_map = feature_map.permute(1, 0, 2).reshape(b, -1, h, w)

        return fused_feature_map, self.linear_projection(feature_map)


class CLIPClassifier(nn.Module):
    """
    GradCAM是根据最终计算得到的Classification Logits反向传播计算的, 因此需要构建一个使用CLIP的Vision Encoder作为特征提取器的分类器

    同时需要注意, 因为要计算GradCAM, 所以分类器必须是可训练的, 不能是冻结的, 因此deepcopy一份出来
    """

    def __init__(
        self, feature_extractor: VisionTransformer, classifier_weights: torch.Tensor
    ):
        super().__init__()

        self.feature_extractor = deepcopy(feature_extractor)
        self.feature_extractor.train()
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.classifier = nn.Linear(*classifier_weights.T.shape, bias=False)
        self.classifier.weight.data.copy_(classifier_weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_feature, _, _ = self.feature_extractor(image)
        logits: torch.Tensor = self.classifier(image_feature)
        return logits.softmax(dim=-1)


class CLIPOutputTarget:
    def __init__(self, category: int):
        self.category = category

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class WeCLIPModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        background_labels: list[str],
        foreground_labels: list[str],
        feature_dim: int = 256,
        num_classes: int = 21,
    ):
        """
        WeCLIP算法中的网络模型

        Args:
            backbone (str): 使用的CLIP的Vision Encoder的backbone. 原论文中是VIT.
            feature_dim (int, optional): Decoder中自注意模块的特征维度. Defaults to 256.
            num_classes (int, optional): 类别数量. Defaults to 21.
        """
        super().__init__()

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.clip: CLIP = clip.load(backbone, device="cpu")[0]
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

        self.patch_size = [
            self.clip.image_resolution // self.clip.vision_patch_size
        ] * 2

        self.feature_decoder = FeatureDecoder(
            num_heads=8,
            num_layers=3,
            mid_dim=feature_dim,
            num_classes=num_classes,
            in_dim=self.clip.vision_width,
            num_features=self.clip.vision_layers,
        )

        self.background_labels = background_labels
        self.foreground_labels = foreground_labels
        self.clip_classifier = CLIPClassifier(
            self.clip.visual,
            torch.cat(
                [self.encode_text(i) for i in [foreground_labels, background_labels]],
                dim=0,
            ),
        )

        self.gradcam = GradCAM(
            model=self.clip_classifier,
            target_layers=[
                self.clip_classifier.feature_extractor.transformer.resblocks[-1].ln_1
            ],
            reshape_transform=self.reshape,
        )

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        return x[1:, :, :].permute(1, 2, 0).reshape(x.shape[1], -1, *self.patch_size)

    def get_gradcams(self, image: torch.Tensor) -> torch.Tensor:
        # [B, num_classes - 1, h, w], -1 是因为GradCAM计算的时候不计算背景类的GradCAM
        # 具体来说是对背景类进行了拆分, 用背景类中的多个物体来表示背景类, 该方法源自 CLIP-ES Fig.3 下面的那段话
        return torch.stack(
            [
                torch.from_numpy(
                    self.gradcam(
                        [image, *self.patch_size],
                        [CLIPOutputTarget(category=i)],
                    )
                )
                for i, _ in enumerate(self.foreground_labels)
            ],
            dim=1,
        ).to(image.device, image.dtype)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # image: [B, 3, 224, 224]
        # feature_map: [B, num_layer, h, w, feature_dim]
        image_feature, feature_map, attention_maps = self.encode_image(image)

        # fused_feature_map: [B, h, w, feature_dim]
        # prediction_mask: [B, num_classes, h, w]
        fused_feature_map: torch.Tensor
        fused_feature_map, prediction_mask = self.feature_decoder(feature_map)

        fused_feature_map = fused_feature_map.flatten(1, -2)
        affinity_map = (
            fused_feature_map @ fused_feature_map.transpose(-2, -1)
        ).sigmoid()

        # image_feature: [B, 512]
        # affinity map: [B, h * w, h * w]
        # prediction_mask: [B, num_classes, h, w]
        # attention maps: [B, num_layer, h * w, h * w]
        return image_feature, affinity_map, prediction_mask, attention_maps

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_image(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_feature: torch.Tensor  # [B, 512]
        patch_feature: torch.Tensor  # [B, num_layer, num_patches + 1, token_dim]
        attention_maps: torch.Tensor  # [B, num_layer, num_patches + 1, num_patches + 1]
        (image_feature, patch_feature, attention_maps) = self.clip.encode_image(image)

        # 注意, 这里去掉了clip的Vision Transformer中在token序列最前面添加的CLS token
        # [B, num_layer, num_patches + 1, token_dim] -> [B, num_layer, num_patches, token_dim]
        patch_feature = torch.stack(patch_feature, dim=1)[:, :, 1:, :]
        # [B, num_layer, num_patches, token_dim] -> [B, num_layer, token_dim, num_patches]
        patch_feature = patch_feature.permute(0, 1, 3, 2)
        # 把token还原为图像矩阵, 即Vision Transformer Patch化的反操作
        # [B, num_layer, token_dim, num_patches] -> [B, num_layer, token_dim, h, w]
        feature_map = patch_feature.reshape(*patch_feature.shape[:3], *self.patch_size)
        # [B, num_layer, token_dim, h, w] -> [B, num_layer, h, w, token_dim]
        feature_map = feature_map.permute(0, 1, 3, 4, 2)

        # 同样去掉CLS token的注意力
        # image_feature: [B, 512]
        # feature_map: [B, num_layer, h, w, token_dim]
        # attention_maps: [B, num_layer, h * w, h * w]
        return (
            image_feature,
            feature_map,
            torch.stack(attention_maps, dim=1)[:, :, 1:, 1:],
        )

    @torch.no_grad()
    def encode_text(
        self,
        text: str | list[str] | torch.Tensor,
        content_length: int = 77,
        truncate: bool = False,
    ) -> torch.Tensor:
        text = (
            text
            if isinstance(text, torch.Tensor)
            else clip.tokenize(text, content_length, truncate)
        )

        return self.clip.encode_text(text)


if __name__ == "__main__":

    import torch.nn.functional as F

    model = WeCLIPModel("ViT-B/32")

    image = torch.randn(2, 3, 224, 224)
    affinity_map, prediction_mask, attention_maps = model(image)
    print(affinity_map.shape)
    print(prediction_mask.shape)
    print(attention_maps.shape)
