# dlphys/init/standard.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from dlphys.config.base import ExperimentConfig
from .registry import register_init


def _iter_named_weight_modules(model: nn.Module):
    """
    Yield (name, module) for modules that have a 'weight' parameter we want to init.
    """
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            yield name, m


def _init_bias(m: nn.Module, *, bias: str = "zero", bias_std: float = 0.0) -> None:
    if getattr(m, "bias", None) is None:
        return
    if bias == "zero":
        nn.init.zeros_(m.bias)
    elif bias == "normal":
        nn.init.normal_(m.bias, mean=0.0, std=float(bias_std))
    else:
        raise ValueError(f"Unknown bias init: {bias}. Use 'zero' or 'normal'.")


def _find_last_linear(model: nn.Module) -> Optional[Tuple[str, nn.Linear]]:
    last: Optional[Tuple[str, nn.Linear]] = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last = (name, m)
    return last


@register_init("identity", overwrite=True)
def init_identity(cfg: ExperimentConfig, model: nn.Module, **kwargs: Any) -> Dict[str, Any]:
    """No-op init, useful for debugging."""
    return {"num_modules": 0}


@register_init("kaiming", overwrite=True)
def init_kaiming(
    cfg: ExperimentConfig,
    model: nn.Module,
    *,
    nonlinearity: str = "relu",
    a: float = 0.0,                 # negative slope for leaky_relu
    mode: str = "fan_in",
    distribution: str = "normal",   # "normal" or "uniform"
    gain: float = 1.0,
    bias: str = "zero",
    bias_std: float = 0.0,
    head_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Kaiming init for Linear/Conv weights + bias init.
    Optionally rescales the last Linear layer by head_scale.
    """
    count = 0
    with torch.no_grad():
        for _, m in _iter_named_weight_modules(model):
            if distribution == "normal":
                nn.init.kaiming_normal_(m.weight, a=float(a), mode=mode, nonlinearity=nonlinearity)
            elif distribution == "uniform":
                nn.init.kaiming_uniform_(m.weight, a=float(a), mode=mode, nonlinearity=nonlinearity)
            else:
                raise ValueError("distribution must be 'normal' or 'uniform'")
            if float(gain) != 1.0:
                m.weight.mul_(float(gain))
            _init_bias(m, bias=bias, bias_std=bias_std)
            count += 1

        head_name = None
        if float(head_scale) != 1.0:
            last = _find_last_linear(model)
            if last is not None:
                head_name, head = last
                head.weight.mul_(float(head_scale))
                if head.bias is not None:
                    head.bias.mul_(float(head_scale))

    return {"num_modules": count, "head_name": head_name, "head_scale": float(head_scale)}


@register_init("xavier", overwrite=True)
def init_xavier(
    cfg: ExperimentConfig,
    model: nn.Module,
    *,
    distribution: str = "uniform",  # "uniform" or "normal"
    gain: float = 1.0,
    bias: str = "zero",
    bias_std: float = 0.0,
    head_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Xavier init for Linear/Conv weights + bias init.
    Optionally rescales the last Linear layer by head_scale.
    """
    count = 0
    with torch.no_grad():
        for _, m in _iter_named_weight_modules(model):
            if distribution == "uniform":
                nn.init.xavier_uniform_(m.weight, gain=float(gain))
            elif distribution == "normal":
                nn.init.xavier_normal_(m.weight, gain=float(gain))
            else:
                raise ValueError("distribution must be 'uniform' or 'normal'")
            _init_bias(m, bias=bias, bias_std=bias_std)
            count += 1

        head_name = None
        if float(head_scale) != 1.0:
            last = _find_last_linear(model)
            if last is not None:
                head_name, head = last
                head.weight.mul_(float(head_scale))
                if head.bias is not None:
                    head.bias.mul_(float(head_scale))

    return {"num_modules": count, "head_name": head_name, "head_scale": float(head_scale)}


@register_init("orthogonal", overwrite=True)
def init_orthogonal(
    cfg: ExperimentConfig,
    model: nn.Module,
    *,
    gain: float = 1.0,
    bias: str = "zero",
    bias_std: float = 0.0,
    head_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Orthogonal init for Linear/Conv weights + bias init.
    Orthogonal works for tensors with 2+ dims.
    """
    count = 0
    with torch.no_grad():
        for _, m in _iter_named_weight_modules(model):
            nn.init.orthogonal_(m.weight, gain=float(gain))
            _init_bias(m, bias=bias, bias_std=bias_std)
            count += 1

        head_name = None
        if float(head_scale) != 1.0:
            last = _find_last_linear(model)
            if last is not None:
                head_name, head = last
                head.weight.mul_(float(head_scale))
                if head.bias is not None:
                    head.bias.mul_(float(head_scale))

    return {"num_modules": count, "head_name": head_name, "head_scale": float(head_scale)}
