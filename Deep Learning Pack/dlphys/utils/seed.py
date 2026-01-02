# dlphys/utils/seed.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set RNG seeds for python, numpy, torch (CPU/CUDA).

    Args:
        seed: Random seed.
        deterministic: If True, try to make CUDA operations deterministic.
            Note: full determinism can reduce performance and isn't guaranteed for all ops.
    """
    # Python / NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch deterministic algorithms (may throw if op has no deterministic impl)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions or environments may not support this cleanly
            pass

        # cuBLAS workspace config helps determinism for some GEMM ops.
        # Ideally this env var should be set before importing torch,
        # but setting here is still helpful in many cases.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(worker_id: int) -> None:
    """
    For DataLoader(num_workers>0): make each worker deterministic.
    Use with DataLoader(worker_init_fn=seed_worker).

    It uses torch.initial_seed() which is already set per worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
