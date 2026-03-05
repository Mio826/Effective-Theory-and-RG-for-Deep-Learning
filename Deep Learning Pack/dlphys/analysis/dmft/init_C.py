# dlphys/analysis/dmft/init_C.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np


Array = np.ndarray


def _normalize_C0(C: Array, target_C0: float = 1.0, eps: float = 1e-12) -> Array:
    """Rescale so that C[0] == target_C0."""
    c0 = float(C[0])
    if abs(c0) < eps:
        raise ValueError("C[0] is ~0; cannot normalize.")
    return C * (target_C0 / c0)


def _clip_bounds(C: Array, c0: float = 1.0) -> Array:
    """Enforce |C(l)| <= C(0) (Cauchy–Schwarz necessary condition)."""
    return np.clip(C, -abs(c0), abs(c0))


def validate_basic(C: Array, atol: float = 1e-8) -> Dict[str, float]:
    """
    Basic necessary (not sufficient) validity checks for a correlation function.
    Returns diagnostics; does NOT guarantee PSD.
    """
    C = np.asarray(C, dtype=float)
    diag = {}
    diag["C0"] = float(C[0])
    diag["max_abs"] = float(np.max(np.abs(C)))
    diag["bound_violation"] = float(max(0.0, diag["max_abs"] - abs(diag["C0"])))
    diag["finite"] = float(np.all(np.isfinite(C)))
    # monotonicity (optional heuristic)
    diag["mono_violations"] = float(np.sum(np.diff(np.abs(C)) > atol))
    return diag


def init_exp(L: int, xi: float = 3.0, C0: float = 1.0) -> Array:
    """C(l) = exp(-l/xi)."""
    ell = np.arange(L + 1, dtype=float)
    C = np.exp(-ell / float(xi))
    C = _normalize_C0(C, C0)
    return _clip_bounds(C, C0)


def init_plateau_exp(L: int, q: float = 0.1, xi: float = 3.0, C0: float = 1.0) -> Array:
    """C(l) = q + (1-q) exp(-l/xi).  Good for testing q>0 basins."""
    if not (0.0 <= q < 1.0):
        raise ValueError("q must be in [0,1).")
    ell = np.arange(L + 1, dtype=float)
    C = q + (1.0 - q) * np.exp(-ell / float(xi))
    C = _normalize_C0(C, C0)
    return _clip_bounds(C, C0)


def init_damped_cos(L: int, xi: float = 5.0, omega: float = np.pi / 4, C0: float = 1.0) -> Array:
    """C(l) = exp(-l/xi) cos(omega l). Useful to probe oscillatory solutions."""
    ell = np.arange(L + 1, dtype=float)
    C = np.exp(-ell / float(xi)) * np.cos(float(omega) * ell)
    C = _normalize_C0(C, C0)
    return _clip_bounds(C, C0)


def init_power_law(L: int, alpha: float = 1.0, ell0: float = 1.0, C0: float = 1.0) -> Array:
    """C(l) = (1 + l/ell0)^(-alpha)."""
    ell = np.arange(L + 1, dtype=float)
    C = (1.0 + ell / float(ell0)) ** (-float(alpha))
    C = _normalize_C0(C, C0)
    return _clip_bounds(C, C0)


def init_gaussian_kernel(L: int, sigma: float = 3.0, C0: float = 1.0) -> Array:
    """C(l) = exp(-(l^2)/(2 sigma^2))."""
    ell = np.arange(L + 1, dtype=float)
    C = np.exp(-(ell**2) / (2.0 * float(sigma) ** 2))
    C = _normalize_C0(C, C0)
    return _clip_bounds(C, C0)


def init_from_spectrum(
    L: int,
    n_freq: int = 256,
    seed: int | None = 0,
    smooth: float = 2.0,
    C0: float = 1.0,
) -> Array:
    """
    Random PSD-safe initializer via nonnegative spectral density.
    Constructs a stationary process with spectrum P(omega) >= 0 on [0, pi],
    then C(l) ~ integral P(omega) cos(omega l) d omega.
    This is a robust way to generate "random but physical" C.

    smooth controls how peaky the spectrum is (larger -> smoother).
    """
    rng = np.random.default_rng(seed)
    w = np.linspace(0.0, np.pi, n_freq, dtype=float)

    # Draw nonnegative spectrum and smooth it a bit
    P = rng.random(n_freq) ** float(smooth)  # >=0
    # Normalize spectrum mass (arbitrary scale; we'll renormalize C[0])
    P = P / (np.trapz(P, w) + 1e-12)

    ell = np.arange(L + 1, dtype=float)
    # Discrete cosine integral approximation
    C = np.array([np.trapz(P * np.cos(w * l), w) for l in ell], dtype=float)

    C = _normalize_C0(C, C0)
    return _clip_bounds(C, C0)


# Convenience registry for notebooks
INIT_REGISTRY: Dict[str, Callable[..., Array]] = {
    "exp": init_exp,
    "plateau_exp": init_plateau_exp,
    "damped_cos": init_damped_cos,
    "power_law": init_power_law,
    "gaussian": init_gaussian_kernel,
    "spectrum": init_from_spectrum,
}