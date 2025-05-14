"""
model.py 定义了ExCEL算法 (Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation) 的网络架构

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : model.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from typing import Optional
from collections.abc import Callable

# Third-Party Library
from kmeans_pytorch import kmeans


# Torch Library
import torch
import torch.nn as nn

# My Library
from . import clip


def tokenize(
    text: str | list[str],
    max_length: Optional[int] = 77,
    truncate: bool = True,
) -> torch.Tensor:
    return clip.tokenize(text, max_length, truncate)


class TextSemanticEnrichment(nn.Module):
    def __init__(
        self,
        topk: int,
        lambda_: float,
        text_encoder: Callable,
        num_attribute_embeddings: int = 112,
        descriptions: Optional[list[str]] = None,
    ):
        super().__init__()

        self.topk = topk
        self.lambda_ = lambda_
        self.text_encoder = text_encoder
        self.num_attribute_embeddings = num_attribute_embeddings

        self.descriptions: list[str] = [] if descriptions is None else descriptions
        if descriptions is not None:
            self.description_embeddings: torch.Tensor
            self.register_buffer(
                "description_embeddings", self.text_encoder(tokenize(descriptions))
            )

            self.cluster_ids: torch.Tensor
            self.cluster_centers: torch.Tensor
            id, centroids = kmeans(
                X=self.description_embeddings,
                num_clusters=num_attribute_embeddings,
            )
            self.register_buffer("cluster_ids", id)
            self.register_buffer("cluster_centers", centroids.clone().detach())

            self.cluster_ids.requires_grad = False
            self.cluster_centers.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        x: str | list[str] | torch.Tensor,
    ) -> torch.Tensor:
        # tc: [batch, feature_dim]
        if isinstance(x, (str, list)):
            tc = self.text_encoder(tokenize(x).to(self.cluster_centers.device))
        elif isinstance(x, torch.Tensor):
            assert (
                tc := x
            ).ndim == 2, "x must be a 2D tensor: [batch, feature_dim] or str, list[str]"
        topk = (tc @ self.cluster_centers.T).topk(k=self.topk, dim=-1)

        # Ac: [batch, topk, feature_dim]
        Ac = self.cluster_centers[topk.indices]

        # scores: [batch, topk]
        scores = torch.softmax(torch.einsum("ijk,ik->ij", Ac, tc), dim=-1)

        # Tc: [batch, feature_dim]
        Tc = self.lambda_ * (scores.unsqueeze(-1) * Ac).sum(dim=1)

        return Tc + tc

    def load_state_dict(
        self, state_dict: dict[str, list[str] | torch.Tensor], strict=True, assign=False
    ) -> None:
        self.topk = state_dict.pop("topk", 5)
        self.lambda_ = state_dict.pop("lambda_", 5)
        self.descriptions = state_dict.pop("descriptions", [])
        super().load_state_dict(state_dict, strict, assign)
        self.cluster_ids.requires_grad = False
        self.cluster_centers.requires_grad = False

    def state_dict(self, *args, **kwargs) -> dict[str, list[str] | torch.Tensor]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["topk"] = self.topk
        state_dict["lambda_"] = self.lambda_
        state_dict["descriptions"] = self.descriptions
        return state_dict


class ExCELModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        topk: int,
        num_attribute_embeddings: int,
        descriptions: Optional[list[str]] = None,
    ):
        super().__init__()

        self.clip = clip.load(backbone)[0]

        self.tse = TextSemanticEnrichment(
            topk=topk,
            lambda_=1,
            descriptions=descriptions,
            text_encoder=self.clip.encode_text,
            num_attribute_embeddings=num_attribute_embeddings,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


if __name__ == "__main__":
    from .helper import get_descriptions, ClassNames

    model = ExCELModel(
        "RN101",
        topk=20,
        num_attribute_embeddings=112,
        descriptions=sum(
            get_descriptions("voc", ClassNames.voc, 20, "deepseek").values(), []
        ),
    )

    model.tse(["cat", "dog"])
    model.clip.encode_image(
        torch.randn(1, 3, 224, 224).to(model.tse.cluster_centers.device)
    )
