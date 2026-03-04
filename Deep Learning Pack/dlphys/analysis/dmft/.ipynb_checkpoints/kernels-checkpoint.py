# dlphys/analysis/dmft/kernels.py
from __future__ import annotations
import numpy as np

def build_toeplitz_from_corr(C: np.ndarray, L: int) -> np.ndarray:
    """
    Given C(l) for l=0..L (TTI, even symmetry often assumed),
    build Toeplitz matrix T_{tau,tau'} = C(|tau-tau'|).
    """
    T = np.empty((L+1, L+1), dtype=np.float64)
    for i in range(L+1):
        for j in range(L+1):
            T[i, j] = C[abs(i-j)]
    return T

def logits_cov_from_C(C: np.ndarray, L: int, eps: float = 1e-12) -> np.ndarray:
    """
    S(C) for logits s in R^{L+1}.
    Under iid Gaussian init and isotropy:
      S_{tau,tau'} = C(0) * C(|tau-tau'|).
    Add small diagonal eps for numerical stability.
    """
    C0 = float(C[0])
    T = build_toeplitz_from_corr(C, L)
    S = C0 * T
    # jitter for Cholesky safety
    S = S + eps * np.eye(L+1, dtype=np.float64)
    return S