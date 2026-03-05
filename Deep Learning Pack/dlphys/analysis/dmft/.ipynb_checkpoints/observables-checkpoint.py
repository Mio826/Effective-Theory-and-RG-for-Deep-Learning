# dlphys/analysis/dmft/observables.py
from __future__ import annotations
import numpy as np

def residual_memory(C: np.ndarray) -> float:
    # for finite L, use the tail value as proxy
    return float(C[-1])

def corr_length_exp(C: np.ndarray, lmin: int = 1, lmax: int | None = None) -> float:
    """
    Fit C(l) ~ exp(-l/xi) on positive lags where C>0.
    """
    if lmax is None:
        lmax = len(C) - 1
    lags = np.arange(lmin, lmax + 1)
    vals = C[lmin:lmax + 1]
    mask = vals > 1e-14
    if np.sum(mask) < 2:
        return float("nan")
    x = lags[mask].astype(np.float64)
    y = np.log(vals[mask].astype(np.float64))
    slope, _ = np.polyfit(x, y, 1)
    if slope >= 0:
        return float("inf")
    return float(-1.0 / slope)

def power_law_alpha(C: np.ndarray, lmin: int = 1, lmax: int | None = None) -> float:
    """
    Fit C(l) ~ l^{-alpha} on positive lags where C>0.
    """
    if lmax is None:
        lmax = len(C) - 1
    lags = np.arange(lmin, lmax + 1)
    vals = C[lmin:lmax + 1]
    mask = (vals > 1e-14) & (lags > 0)
    if np.sum(mask) < 2:
        return float("nan")
    x = np.log(lags[mask].astype(np.float64))
    y = np.log(vals[mask].astype(np.float64))
    slope, _ = np.polyfit(x, y, 1)
    return float(-slope)

def focus_sharpness(M: Array) -> float:
    """
    Dimensionless sharpness:
        kappa = (L+1) * E[||alpha||_2^2] = (L+1) * trace(M)

    Range:
      - Uniform attention: kappa = 1
      - One-hot attention: kappa ~ L+1
    """
    M = np.asarray(M, dtype=float)
    Lp1 = M.shape[0]
    return float(Lp1 * np.trace(M))