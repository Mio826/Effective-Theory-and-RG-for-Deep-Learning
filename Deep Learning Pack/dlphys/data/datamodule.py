# dlphys/data/datamodule.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from dlphys.utils.seed import seed_worker


@dataclass
class DataLoaders:
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None


def make_split(
    ds: Dataset,
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[Dataset, Optional[Dataset]]:
    if val_fraction <= 0:
        return ds, None
    n = len(ds)
    n_val = int(round(n * val_fraction))
    n_train = n - n_val
    g = torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    return train_ds, val_ds


def make_loader(
    ds: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    g = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
        generator=g,
    )
