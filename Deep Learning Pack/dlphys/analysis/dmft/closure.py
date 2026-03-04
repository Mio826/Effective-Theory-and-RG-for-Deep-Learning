# dlphys/analysis/dmft/closure.py
from __future__ import annotations
import numpy as np

def corr_get(C: np.ndarray, lag: int) -> float:
    """
    C is defined for l=0..L.
    Use even symmetry for negative lag: C(-l)=C(l).
    Truncate for l>L: return 0.
    """
    if lag < 0:
        lag = -lag
    if lag >= len(C):
        return 0.0
    return float(C[lag])

def dmft_map_C(
    C: np.ndarray,    # shape (L+1,)
    M: np.ndarray,    # shape (L+1,L+1)
    L: int
) -> np.ndarray:
    """
    C_new(l) = sum_{tau,tau'} M_{tau,tau'} * C(l+tau-tau')
    for l=0..L.
    """
    Lp1 = L + 1
    Cnew = np.zeros((Lp1,), dtype=np.float64)
    for l in range(Lp1):
        acc = 0.0
        for tau in range(Lp1):
            for tau2 in range(Lp1):
                acc += M[tau, tau2] * corr_get(C, l + tau - tau2)
        Cnew[l] = acc
    return Cnew