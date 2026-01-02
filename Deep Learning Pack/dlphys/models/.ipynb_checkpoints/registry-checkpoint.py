# dlphys/models/registry.py
from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from dlphys.config.base import ExperimentConfig
from dlphys.config.registry import register_model

from .mlp import MLP, MLPConfig
from .convnet import ConvNetSmall, ConvNetSmallConfig
from .resnet import resnet18_cifar


@register_model("mlp", overwrite=True)
def build_mlp(cfg: ExperimentConfig, **kwargs: Any) -> nn.Module:
    # required: in_dim, out_dim
    if "in_dim" not in kwargs or "out_dim" not in kwargs:
        raise ValueError("mlp requires model_kwargs: {in_dim: int, out_dim: int, ...}")

    mc = MLPConfig(
        in_dim=int(kwargs["in_dim"]),
        out_dim=int(kwargs["out_dim"]),
        width=int(kwargs.get("width", 256)),
        depth=int(kwargs.get("depth", 4)),
        activation=str(kwargs.get("activation", "relu")),
        norm=str(kwargs.get("norm", "none")),
        dropout=float(kwargs.get("dropout", 0.0)),
        bias=bool(kwargs.get("bias", True)),
        residual=bool(kwargs.get("residual", False)),
    )
    return MLP(mc)


@register_model("convnet_small", overwrite=True)
def build_convnet_small(cfg: ExperimentConfig, **kwargs: Any) -> nn.Module:
    cc = ConvNetSmallConfig(
        in_channels=int(kwargs.get("in_channels", 3)),
        num_classes=int(kwargs.get("num_classes", 10)),
        width=int(kwargs.get("width", 64)),
        depth=int(kwargs.get("depth", 3)),
        norm=str(kwargs.get("norm", "batchnorm")),
        dropout=float(kwargs.get("dropout", 0.0)),
    )
    return ConvNetSmall(cc)


@register_model("resnet18_cifar", overwrite=True)
def build_resnet18_cifar(cfg: ExperimentConfig, **kwargs: Any) -> nn.Module:
    # norm: "batchnorm" | "none"
    return resnet18_cifar(
        num_classes=int(kwargs.get("num_classes", 10)),
        norm=str(kwargs.get("norm", "batchnorm")),
        width=int(kwargs.get("width", 64)),
        in_channels=int(kwargs.get("in_channels", 3)),
    )


# Convenience aliases for ablation
@register_model("resnet18_cifar_nobn", overwrite=True)
def build_resnet18_cifar_nobn(cfg: ExperimentConfig, **kwargs: Any) -> nn.Module:
    kwargs = dict(kwargs)
    kwargs["norm"] = "none"
    return build_resnet18_cifar(cfg, **kwargs)
