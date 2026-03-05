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


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def estimate_attention_entropy_from_S(
    S: np.ndarray,
    n_mc: int = 50000,
    seed: int | None = 0,
    batch: int = 2000,
    eps: float = 1e-12,
) -> float:
    """
    Estimate H = E[ -sum_tau alpha_tau log alpha_tau ],
    where alpha = softmax(s), s ~ N(0, S).

    Uses the same Gaussian sampling pipeline as M(C), but returns scalar entropy.
    """
    rng = np.random.default_rng(seed)
    Lp1 = S.shape[0]

    # Robust-ish: symmetrize then add tiny jitter
    Ssym = 0.5 * (S + S.T)
    jitter = 1e-12 * np.eye(Lp1)
    Lchol = np.linalg.cholesky(Ssym + jitter)

    done = 0
    H_acc = 0.0
    while done < n_mc:
        b = min(batch, n_mc - done)
        z = rng.standard_normal((b, Lp1))
        s = z @ Lchol.T
        a = softmax(s, axis=-1)
        H = -np.sum(a * np.log(a + eps), axis=-1)  # (b,)
        H_acc += float(np.sum(H))
        done += b

    return H_acc / float(n_mc)


def entropy_normalized(H: float, L: int) -> float:
    """
    Normalize entropy to [0,1] by dividing log(L+1).
    1: uniform, 0: one-hot (in the large-s limit).
    """
    return float(H / np.log(L + 1))