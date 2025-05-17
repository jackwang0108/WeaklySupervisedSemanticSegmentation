"""
CAM系列方法的核心对象, 通过ActivationsAndGradients类包裹了目标层, 并为其添加forward和backward hook
从而实现了对中间层激活值和梯度的提取, 进而用于支持计算各种CAM

版权归原作者所有, 这里对原作者的代码进行了部分修改, 例如添加类型注释等等, 从而使其更加符合现代Python的编程风格

    @Time    : 2025/05/17
    @Author  : JackWang
    @File    : activations_and_gradients.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
from collections.abc import Callable

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

# My Library


class ActivationsAndGradients:
    """Class for extracting activations and
    registering gradients from targeted intermediate layers"""

    def __init__(
        self,
        model: nn.Module,
        target_layers: list[nn.Module],
        reshape_transform: Callable[[torch.Tensor], torch.Tensor],
        detach: bool = True,
    ):
        self.model: nn.Module = model
        self.gradients: list[torch.Tensor] = []
        self.activations: list[torch.Tensor] = []
        self.reshape_transform: Callable[[torch.Tensor], torch.Tensor] = (
            reshape_transform
        )
        self.detach = detach
        self.handles: list[RemovableHandle] = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ):
        activation = output
        if self.detach:
            if self.reshape_transform is not None:
                activation = self.reshape_transform(activation)
            self.activations.append(activation.cpu().detach())
        else:
            self.activations.append(activation)

    def save_gradient(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad: torch.Tensor):
            if self.detach:
                if self.reshape_transform is not None:
                    grad = self.reshape_transform(grad)
                self.gradients = [grad.cpu().detach()] + self.gradients
            else:
                self.gradients = [grad] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
