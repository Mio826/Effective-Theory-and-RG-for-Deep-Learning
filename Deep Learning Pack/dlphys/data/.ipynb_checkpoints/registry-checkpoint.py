# dlphys/data/registry.py
from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import TensorDataset

from dlphys.config.base import ExperimentConfig
from dlphys.config.registry import register_data

from .datamodule import DataLoaders, make_loader, make_split


@register_data("toy_linear_regression", overwrite=True)
def build_toy_linear_regression(
    cfg: ExperimentConfig,
    *,
    n: int = 512,
    in_dim: int = 4,
    out_dim: int = 3,
    noise_std: float = 0.1,
    val_fraction: float = 0.0,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoaders:
    """
    Synthetic regression:
      x ~ N(0,1)
      y = x W + eps

    This is the minimal data pipeline smoke-test (no external deps).
    """
    # Use cfg seed for full reproducibility
    torch.manual_seed(int(cfg.seed))

    x = torch.randn(int(n), int(in_dim))
    W = torch.randn(int(in_dim), int(out_dim))
    y = x @ W + float(noise_std) * torch.randn(int(n), int(out_dim))

    ds = TensorDataset(x, y)
    train_ds, val_ds = make_split(ds, val_fraction=float(val_fraction), seed=cfg.seed)

    bs = int(batch_size) if batch_size is not None else int(cfg.batch_size)

    train_loader = make_loader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(num_workers),
        seed=cfg.seed,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = make_loader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=int(num_workers),
            seed=cfg.seed + 1,
            drop_last=False,
        )

    return DataLoaders(train_loader=train_loader, val_loader=val_loader, test_loader=None)
