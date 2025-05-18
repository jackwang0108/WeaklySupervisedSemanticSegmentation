"""
cam_refinement.py 定义了对 Grad-CAM 进行细化的函数 cam_refinement

对应论文中的 Sec.3.3 Frozen CLIP CAM Refinement

    @Time    : 2025/05/17
    @Author  : JackWang
    @File    : cam_refinement.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library

# Third-Party Library
import cv2
import numpy as np

# Torch Library
import torch

# My Library


def sinkhorn_normalization(
    R: torch.Tensor, sinkhorn_iters: int = 2, power_iters: int = 1
) -> torch.Tensor:
    # Sinkhorn 归一化
    for _ in range(sinkhorn_iters):
        R = R / (R.sum(dim=3, keepdim=True) + 1e-8)  # 行归一化
        R = R / (R.sum(dim=2, keepdim=True) + 1e-8)  # 列归一化

    # 对称化
    R = (R + R.transpose(2, 3)) / 2

    # 幂迭代
    for _ in range(power_iters):
        R = torch.bmm(R.squeeze(), R.squeeze())

    return R.unsqueeze(1)


_CONTOUR_INDEX = 1 if cv2.__version__.split(".")[0] == "3" else 0


def scoremap2bbox(
    scoremap: np.ndarray, threshold: float, multi_contour_eval: bool = False
):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY,
    )
    contours = cv2.findContours(
        image=thr_gray_heatmap, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)


def generate_mask(cam: torch.Tensor, threshold: float) -> torch.Tensor:
    masks: list[torch.Tensor] = []
    device: torch.device = cam.device
    B, C, H, W = cam.shape
    for b in range(B):
        batch_masks: list[torch.Tensor] = []
        for c in range(C):
            cam_np = cam[b, c].detach().cpu().numpy()
            boxes, _ = scoremap2bbox(cam_np, threshold)
            mask = np.zeros((H, W), dtype=np.float32)
            for box in boxes:
                x0, y0, x1, y1 = box
                mask[y0 : y1 + 1, x0 : x1 + 1] = 1
            batch_masks.append(torch.from_numpy(mask).flatten())
        masks.append(torch.stack(batch_masks))
    return torch.stack(masks).to(device)


def cam_refinement(
    cam: torch.Tensor,
    affinity_map: torch.Tensor,
    attention_maps: torch.Tensor,
    n0: int,
    alpha: float,
) -> torch.Tensor:
    """
    该函数对应原始论文中的Frozen CLIP CAM Refinement Module, 本质上是用CLIP的Visual Encoder中的Attention Map对原始CAM进行增强, 从而得到训练目标

    WeCLIP这里是在CLIP-ES的CAA模块上改进得来的, 详细原理需要去看CLIP-ES的论文 Sec. 3.3 Class-aware Attention-based Affinity (CAA)
    """
    # 原文中的 Eq 5
    # Sl: [B, 12, 1, 1]
    Sl = (
        (affinity_map.unsqueeze(dim=1) - attention_maps)
        .abs()
        .sum(dim=(-2, -1), keepdim=True)
    )

    # 原文中的 Eq 6
    # threshold: [B, 1, 1, 1]
    # Gl: [B, 12, 1, 1]
    threshold = Sl[:, n0:].mean(dim=1, keepdim=True)
    Gl = (Sl < threshold).to(cam.dtype)

    # 原文中的 Eq 7
    # Nm: [B, 1, 1, 1]
    # R: [B, 1, h * w, h * w]
    Nm = Gl[:, :n0, :, :].sum(dim=1, keepdim=True)
    R = (
        affinity_map.unsqueeze(dim=1)
        / Nm
        * (attention_maps * Gl).sum(dim=1, keepdim=True)
    )

    mask = generate_mask(cam, 0.4)
    trans = sinkhorn_normalization(R).squeeze() ** alpha
    masked_trans = trans.unsqueeze(1) * mask.unsqueeze(-1)

    # 原文中的 Eq 8
    # refined_cam: [B, num_class, h * w]
    Mf = torch.matmul(masked_trans, cam.flatten(2).unsqueeze(-1)).squeeze(-1)
    return Mf.view_as(cam)
