# dlphys/analysis/dmft/solver.py
from __future__ import annotations
import numpy as np

from .kernels import logits_cov_from_C
from .softmax_moments import estimate_M_from_S
from .closure import dmft_map_C

class DMFTConfig:
    def __init__(
        self,
        L: int,
        gamma: float,
        n_mc: int = 20000,
        mc_seed: int = 0,
        tol: float = 1e-7,
        max_iter: int = 500,
        damping: float = 0.2,   # C <- (1-a)C + a F(C)
        jitter: float = 1e-12,
        mc_batch: int = 2000,
    ):
        self.L = int(L)
        self.gamma = float(gamma)
        self.n_mc = int(n_mc)
        self.mc_seed = int(mc_seed)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.damping = float(damping)
        self.jitter = float(jitter)
        self.mc_batch = int(mc_batch)

class DMFTSolver:
    """
    Implements: C -> S(C) -> M(C) -> F(C).
    Notes:
      - gamma enters ONLY through the logits scaling in the *full* theory.
        In the simplified S(C)=C0*C(|tau-tau'|) model, gamma does not appear.
      - If you want gamma-dependence, you should incorporate gamma as:
          s <- s / d_k^(gamma - 1/2)  (or equivalent rescaling),
        which in the GP level amounts to scaling S(C) by a factor.
    """
    def __init__(self, cfg: DMFTConfig):
        self.cfg = cfg

    def compute_S(self, C: np.ndarray) -> np.ndarray:
        S = logits_cov_from_C(C, self.cfg.L, eps=self.cfg.jitter)
        # ---- gamma hook (minimal): rescale logits variance ----
        # In many conventions, standard transformer uses 1/sqrt(dk) (gamma=1/2).
        # If your theory uses 1/dk^gamma, then relative scaling is dk^{-(gamma-1/2)}.
        # At GP level: S -> s2 * S for some scalar s2(gamma, dk).
        # Here we leave dk out; you can inject an effective scalar if you decide.
        scale = dk ** (1 - 2 * gamma)
        S *= scale
        return S

    def compute_M(self, S: np.ndarray) -> np.ndarray:
        return estimate_M_from_S(
            S,
            n_mc=self.cfg.n_mc,
            seed=self.cfg.mc_seed,
            batch=self.cfg.mc_batch,
        )

    def F(self, C: np.ndarray) -> tuple[np.ndarray, dict]:
        S = self.compute_S(C)
        M = self.compute_M(S)
        Cnew = dmft_map_C(C, M, self.cfg.L)
        aux = {"S": S, "M": M}
        return Cnew, aux

    def iterate(self, C0: np.ndarray, verbose: bool = False) -> dict:
        C = C0.astype(np.float64).copy()
        hist = []
        aux_last = None

        for it in range(self.cfg.max_iter):
            Cnew, aux = self.F(C)
            aux_last = aux
            # damping
            a = self.cfg.damping
            Cupd = (1 - a) * C + a * Cnew

            err = float(np.max(np.abs(Cupd - C)))
            hist.append(err)
            if verbose and (it % 10 == 0 or it == self.cfg.max_iter - 1):
                print(f"[DMFT] it={it:04d} err={err:.3e} C0={Cupd[0]:.6f} C1={Cupd[1] if len(Cupd)>1 else np.nan:.6f}")

            C = Cupd
            if err < self.cfg.tol:
                return {
                    "converged": True,
                    "C_star": C,
                    "n_iter": it + 1,
                    "err_hist": np.array(hist),
                    "aux": aux_last,
                }

        return {
            "converged": False,
            "C_star": C,
            "n_iter": self.cfg.max_iter,
            "err_hist": np.array(hist),
            "aux": aux_last,
        }