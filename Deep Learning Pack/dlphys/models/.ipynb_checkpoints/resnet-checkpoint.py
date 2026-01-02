# dlphys/models/resnet.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
        while c % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, c)
    raise ValueError(f"Unknown norm kind: {kind}")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, *, stride: int, norm: str):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = make_norm_2d(norm, planes)
        self.act1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = make_norm_2d(norm, planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

        self.act_out = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + self.shortcut(x)
        out = self.act_out(out)
        return out


@dataclass
class ResNetCIFARConfig:
    num_classes: int = 10
    norm: str = "batchnorm"          # "batchnorm" | "groupnorm" | "none"
    width: int = 64                  # base channel count
    layers: Tuple[int, int, int, int] = (2, 2, 2, 2)  # ResNet-18
    in_channels: int = 3


class ResNetCIFAR(nn.Module):
    """
    CIFAR-style ResNet (no 7x7 conv, no maxpool).
    norm can be "batchnorm" or "none" (for your BN vs no-BN ablation).

    Hook points:
      stem.conv / stem.norm / layer{1-4}.{i}.conv1 etc.
    """

    def __init__(self, cfg: ResNetCIFARConfig):
        super().__init__()
        self.cfg = cfg
        norm = cfg.norm
        w = cfg.width

        self.stem_conv = nn.Conv2d(cfg.in_channels, w, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_norm = make_norm_2d(norm, w)
        self.stem_act = nn.ReLU(inplace=False)

        self.in_planes = w
        self.layer1 = self._make_layer(w,  cfg.layers[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(2*w, cfg.layers[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(4*w, cfg.layers[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(8*w, cfg.layers[3], stride=2, norm=norm)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8*w * BasicBlock.expansion, cfg.num_classes)

    def _make_layer(self, planes: int, num_blocks: int, *, stride: int, norm: str) -> nn.Sequential:
        blocks: List[nn.Module] = []
        strides = [stride] + [1] * (num_blocks - 1)
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, stride=s, norm=norm))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        feats: Dict[str, torch.Tensor] = {}

        out = self.stem_conv(x)
        out = self.stem_norm(out)
        out = self.stem_act(out)
        if return_features:
            feats["stem.out"] = out

        out = self.layer1(out)
        if return_features:
            feats["layer1.out"] = out
        out = self.layer2(out)
        if return_features:
            feats["layer2.out"] = out
        out = self.layer3(out)
        if return_features:
            feats["layer3.out"] = out
        out = self.layer4(out)
        if return_features:
            feats["layer4.out"] = out

        out = self.pool(out)
        out = torch.flatten(out, 1)
        logits = self.fc(out)

        if return_features:
            feats["logits"] = logits
            return logits, feats
        return logits


def resnet18_cifar(*, num_classes: int = 10, norm: str = "batchnorm", width: int = 64, in_channels: int = 3) -> ResNetCIFAR:
    cfg = ResNetCIFARConfig(num_classes=num_classes, norm=norm, width=width, in_channels=in_channels, layers=(2, 2, 2, 2))
    return ResNetCIFAR(cfg)
