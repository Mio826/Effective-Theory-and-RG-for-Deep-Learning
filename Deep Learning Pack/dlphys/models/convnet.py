# dlphys/models/convnet.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


def make_norm_2d(kind: str, c: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind in ("none", "identity"):
        return nn.Identity()
    if kind in ("batchnorm", "bn"):
        return nn.BatchNorm2d(c)
    if kind in ("groupnorm", "gn"):
        g = min(32, c)
        # ensure divisible
        while c % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, c)
    raise ValueError(f"Unknown norm kind: {kind}")


@dataclass
class ConvNetSmallConfig:
    in_channels: int = 3
    num_classes: int = 10
    width: int = 64
    depth: int = 3
    norm: str = "batchnorm"    # "batchnorm" | "groupnorm" | "none"
    dropout: float = 0.0


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, *, norm: str, dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = make_norm_2d(norm, c_out)
        self.act = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class ConvNetSmall(nn.Module):
    """
    Simple ConvNet for MNIST/CIFAR baselines.

    Hook points:
      blocks.{i}.conv / blocks.{i}.norm / blocks.{i}.act
    """

    def __init__(self, cfg: ConvNetSmallConfig):
        super().__init__()
        self.cfg = cfg

        blocks = []
        c = cfg.in_channels
        for i in range(cfg.depth):
            blocks.append(ConvBlock(c, cfg.width, norm=cfg.norm, dropout=cfg.dropout))
            c = cfg.width
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        feats: Dict[str, torch.Tensor] = {}

        h = x
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if return_features:
                feats[f"block{i}.out"] = h

        logits = self.head(h)
        if return_features:
            feats["logits"] = logits
            return logits, feats
        return logits
