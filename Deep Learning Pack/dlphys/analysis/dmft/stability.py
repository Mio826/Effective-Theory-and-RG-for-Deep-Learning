# dlphys/analysis/dmft/stability.py
from __future__ import annotations
import numpy as np

def finite_diff_jacobian(F_only, C_star: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Build Jacobian J_{ij} = d F_i / d C_j at C_star by finite differences.
    F_only: function C -> Cnew (no aux).
    """
    n = len(C_star)
    J = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        d = np.zeros((n,), dtype=np.float64)
        d[j] = eps
        fp = F_only(C_star + d)
        fm = F_only(C_star - d)
        J[:, j] = (fp - fm) / (2 * eps)
    return J

def spectral_radius(J: np.ndarray) -> float:
    vals = np.linalg.eigvals(J)
    return float(np.max(np.abs(vals)))