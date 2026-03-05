# dlphys/analysis/dmft/classify_phase.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PhaseReport:
    phase: str
    q_hat: float
    xi_hat: float | None
    alpha_hat: float | None
    fit_exp_r2: float | None
    fit_pow_r2: float | None
    notes: str


def _linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit y = a x + b. Return (a, b, R^2).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-15
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2)


def classify_phase(
    C: np.ndarray,
    tail_frac: float = 0.25,
    q_tol: float = 1e-3,
    fit_min_l: int = 1,
    fit_max_l: int | None = None,
    eps: float = 1e-12,
) -> PhaseReport:
    """
    Classify a converged correlation function C(l), l=0..L.

    Labels:
      - "plateau"  : q_hat > q_tol
      - "exp"      : exponential fit better than power-law
      - "powerlaw" : power-law fit better than exponential
      - "unclear"  : neither fit good

    Uses simple regression on log(C(l) - q_hat) or log(C(l)).
    """
    C = np.asarray(C, float)
    L = len(C) - 1
    if fit_max_l is None:
        fit_max_l = L

    # --- 1) estimate plateau q from tail ---
    tail_len = max(2, int(np.ceil(tail_frac * (L + 1))))
    tail = C[-tail_len:]
    q_hat = float(np.mean(tail))

    # If plateau is significant, subtract it for decay fitting
    if q_hat > q_tol:
        Cdec = C - q_hat
        notes0 = "tail plateau detected; fitting decay of C-q."
    else:
        Cdec = C.copy()
        notes0 = "no significant plateau; fitting decay of C."

    # choose fitting window
    l = np.arange(L + 1)
    lo = max(fit_min_l, 1)           # avoid l=0 for logs
    hi = min(fit_max_l, L)
    lfit = l[lo:hi + 1]
    yfit = Cdec[lo:hi + 1]

    # keep only positive entries for log fits
    mask = yfit > eps
    lfit = lfit[mask]
    yfit = yfit[mask]

    if len(lfit) < 3:
        return PhaseReport(
            phase="unclear",
            q_hat=q_hat,
            xi_hat=None,
            alpha_hat=None,
            fit_exp_r2=None,
            fit_pow_r2=None,
            notes=notes0 + " not enough positive points to fit.",
        )

    # --- 2) exponential fit: log y = a l + b -> xi = -1/a ---
    a_exp, b_exp, r2_exp = _linear_fit_r2(lfit, np.log(yfit))
    xi_hat = None
    if a_exp < -eps:
        xi_hat = float(-1.0 / a_exp)

    # --- 3) power-law fit: log y = -alpha log l + b ---
    a_pow, b_pow, r2_pow = _linear_fit_r2(np.log(lfit), np.log(yfit))
    alpha_hat = float(-a_pow)

    # --- 4) decide label ---
    # Use R^2 and some sanity thresholds
    best = "unclear"
    notes1 = f"R2_exp={r2_exp:.4f}, R2_pow={r2_pow:.4f}."
    if q_hat > q_tol:
        # plateau dominates long-range; still can be exp/power in transient
        best = "plateau"
        notes = notes0 + " " + notes1
        return PhaseReport(best, q_hat, xi_hat, alpha_hat, r2_exp, r2_pow, notes)

    # if not plateau, compare fits
    if (r2_exp > 0.98) and (r2_exp >= r2_pow + 0.01):
        best = "exp"
    elif (r2_pow > 0.98) and (r2_pow >= r2_exp + 0.01):
        best = "powerlaw"
    else:
        # pick the better one but mark unclear if both mediocre
        if r2_exp >= r2_pow and r2_exp > 0.9:
            best = "exp-ish"
        elif r2_pow > r2_exp and r2_pow > 0.9:
            best = "powerlaw-ish"
        else:
            best = "unclear"

    notes = notes0 + " " + notes1
    return PhaseReport(best, q_hat, xi_hat, alpha_hat, r2_exp, r2_pow, notes)