# dlphys/analysis/dmft/softmax_moments.py
from __future__ import annotations
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def estimate_M_from_S(
    S: np.ndarray,
    n_mc: int = 20000,
    seed: int | None = 0,
    batch: int = 2000,
) -> np.ndarray:
    """
    M_{tau,tau'} = E[ alpha_tau alpha_tau' ],  alpha = softmax(s), s~N(0,S).
    Returns (L+1, L+1) matrix.

    batch sampling to reduce memory.
    """
    rng = np.random.default_rng(seed)
    Lp1 = S.shape[0]
    # Cholesky for Gaussian sampling
    Lchol = np.linalg.cholesky(S)

    M = np.zeros((Lp1, Lp1), dtype=np.float64)
    done = 0
    while done < n_mc:
        b = min(batch, n_mc - done)
        z = rng.standard_normal((b, Lp1))
        s = z @ Lchol.T
        a = softmax(s, axis=-1)  # (b, L+1)
        # accumulate E[a^T a]
        M += (a.T @ a)
        done += b
    M /= float(n_mc)
    return M