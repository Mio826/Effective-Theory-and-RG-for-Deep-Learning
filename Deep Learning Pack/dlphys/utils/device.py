# dlphys/utils/device.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple, Union

import torch


def get_device(device: str = "cuda") -> torch.device:
    """
    Resolve a torch.device with safe fallback.
    - "cuda" -> cuda if available else cpu
    - "cpu" -> cpu
    - "cuda:0" -> that device if available else cpu
    """
    device = (device or "cuda").lower()

    if device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device)
        return torch.device("cpu")

    if device == "mps":
        # For Apple Silicon; safe fallback on non-mac.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")


def to_device(batch: Any, device: torch.device) -> Any:
    """
    Recursively move tensors in common batch structures to device.
    Supports:
      - torch.Tensor
      - (x, y, ...)
      - list [...]
      - dict {k: v}
    """
    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, tuple):
        return tuple(to_device(x, device) for x in batch)

    if isinstance(batch, list):
        return [to_device(x, device) for x in batch]

    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}

    # Unknown type: return as-is (e.g., strings, ints)
    return batch
