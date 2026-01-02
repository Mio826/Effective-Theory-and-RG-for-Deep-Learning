# dlphys/models/mlp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in ("relu",):
        return nn.ReLU(inplace=False)
    if name in ("gelu",):
        return nn.GELU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("sigmoid",):
        return nn.Sigmoid()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name in ("identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def make_norm_1d(kind: str, dim: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind in ("none", "identity"):
        return nn.Identity()
    if kind in ("layernorm", "ln"):
        return nn.LayerNorm(dim)
    if kind in ("batchnorm", "bn"):
        # Works best when input is (N, dim)
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unknown norm kind: {kind}")


@dataclass
class MLPConfig:
    in_dim: int
    out_dim: int
    width: int = 256
    depth: int = 4                  # number of hidden layers
    activation: str = "relu"
    norm: str = "none"              # "none" | "layernorm" | "batchnorm"
    dropout: float = 0.0
    bias: bool = True
    residual: bool = False          # residual MLP blocks (only meaningful if width constant)


class MLPBlock(nn.Module):
    """
    A block intentionally structured as:
      Linear -> Norm -> Activation -> Dropout
    so you can hook:
      - preact: output of Linear
      - postnorm: output of Norm (if used)
      - act: output of Activation
    """

    def __init__(self, dim_in: int, dim_out: int, *, activation: str, norm: str, dropout: float, bias: bool):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.norm = make_norm_1d(norm, dim_out)
        self.act = make_activation(activation)
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MLP(nn.Module):
    """
    MLP with stable module names for hooking:
      blocks.{i}.linear / blocks.{i}.norm / blocks.{i}.act

    forward(x) -> logits
    forward(x, return_features=True) -> (logits, features_dict)
    """

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg

        blocks = []
        dim = cfg.in_dim
        for i in range(cfg.depth):
            blocks.append(
                MLPBlock(
                    dim_in=dim,
                    dim_out=cfg.width,
                    activation=cfg.activation,
                    norm=cfg.norm,
                    dropout=cfg.dropout,
                    bias=cfg.bias,
                )
            )
            dim = cfg.width
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(dim, cfg.out_dim, bias=cfg.bias)

        self.residual = bool(cfg.residual) and (cfg.in_dim == cfg.width)

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        feats: Dict[str, torch.Tensor] = {}

        h = x
        for i, blk in enumerate(self.blocks):
            h_in = h
            h = blk(h)
            if self.residual and h.shape == h_in.shape:
                h = h + h_in
            if return_features:
                feats[f"block{i}.out"] = h

        logits = self.head(h)
        if return_features:
            feats["logits"] = logits
            return logits, feats
        return logits
