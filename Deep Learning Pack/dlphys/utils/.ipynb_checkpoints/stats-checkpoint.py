# dlphys/utils/stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


ArrayLike = Union[float, int, np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class RunningMeanVar:
    """
    Online mean/variance using Welford algorithm.

    Supports scalar or array statistics.
    - If you feed arrays of same shape, it tracks elementwise mean/var.
    - For activations, you typically pass x.reshape(-1) to track scalar mean/var.

    Usage:
        r = RunningMeanVar()
        r.update(x)
        print(r.mean, r.var, r.std)
    """
    count: int = 0
    mean: Optional[np.ndarray] = None
    M2: Optional[np.ndarray] = None

    def update(self, x: ArrayLike) -> None:
        x = _to_numpy(x).astype(np.float64)

        if self.mean is None:
            self.mean = np.zeros_like(x, dtype=np.float64)
            self.M2 = np.zeros_like(x, dtype=np.float64)

        self.count += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = x - self.mean
        self.M2 = self.M2 + delta * delta2

    @property
    def var(self) -> Optional[np.ndarray]:
        if self.count < 2 or self.M2 is None:
            return None
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> Optional[np.ndarray]:
        v = self.var
        if v is None:
            return None
        return np.sqrt(v)

    def reset(self) -> None:
        self.count = 0
        self.mean = None
        self.M2 = None


@dataclass
class RunningMoments:
    """
    Online moments up to 4th (mean, var, skewness, kurtosis).
    This uses a simple accumulate-on-flatten strategy:
    update with 1D values; internal uses RunningMeanVar + sums.

    For many research metrics, mean/var are enough.
    Skew/kurt are optional but useful for non-Gaussianity tracking.
    """
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    M3: float = 0.0
    M4: float = 0.0

    def update(self, x: ArrayLike) -> None:
        a = _to_numpy(x).astype(np.float64).reshape(-1)
        for xi in a:
            self._update_one(float(xi))

    def _update_one(self, x: float) -> None:
        n1 = self.n
        self.n += 1
        delta = x - self.mean
        delta_n = delta / self.n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1

        self.mean += delta_n
        self.M4 += (
            term1 * delta_n2 * (self.n * self.n - 3 * self.n + 3)
            + 6 * delta_n2 * self.M2
            - 4 * delta_n * self.M3
        )
        self.M3 += term1 * delta_n * (self.n - 2) - 3 * delta_n * self.M2
        self.M2 += term1

    @property
    def var(self) -> Optional[float]:
        if self.n < 2:
            return None
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> Optional[float]:
        v = self.var
        if v is None:
            return None
        return float(np.sqrt(v))

    @property
    def skew(self) -> Optional[float]:
        if self.n < 3:
            return None
        v = self.var
        if v is None or v == 0:
            return None
        return float(np.sqrt(self.n) * self.M3 / (self.M2 ** 1.5))

    @property
    def kurtosis_excess(self) -> Optional[float]:
        if self.n < 4:
            return None
        v = self.var
        if v is None or v == 0:
            return None
        return float(self.n * self.M4 / (self.M2 * self.M2) - 3.0)

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0
