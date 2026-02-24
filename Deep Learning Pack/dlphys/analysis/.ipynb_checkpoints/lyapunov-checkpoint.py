# dlphys/analysis/lyapunov.py
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple

import torch
from .jvp import jvp_F


@torch.no_grad()
def _normalize(v: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize per-sample (batch-wise) vectors.
    v: [B, ...]
    returns: (v_unit, norm) where norm is [B]
    """
    B = v.shape[0]
    flat = v.reshape(B, -1)
    n = torch.linalg.norm(flat, dim=1) + eps
    v_unit = v / n.reshape(B, *([1] * (v.ndim - 1)))
    return v_unit, n


def lyapunov_max_benettin(
    F: Callable[[torch.Tensor], torch.Tensor],
    s0: torch.Tensor,
    *,
    T: int,
    burn_in: int = 0,
    v0: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    return_traj: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Max Lyapunov exponent via Benettin algorithm using JVP:
        s_{t+1} = F(s_t)
        v_{t+1} = J_t v_t
        lambda ≈ mean_{t in window} log ||v_{t+1}||  (with renormalization each step)

    IMPORTANT:
    - This estimates the maximal exponent (top-1).
    - burn_in steps are discarded from averaging.

    Args:
        F: map s -> s_next (should be deterministic)
        s0: initial state, shape [B, ...]
        T: number of steps used for averaging (after burn_in)
        burn_in: steps to discard before averaging
        v0: initial tangent vector, same shape as s0. If None, random N(0,1).
        return_traj: if True, also return final state and some logs.

    Returns:
        dict with:
            lambda_hat: [B] tensor (per-trajectory estimate)
            lambda_mean: scalar tensor (mean over batch)
            final_state: [B,...] (if return_traj)
            logs: [T] or [burn_in+T] (optional)
    """
    if s0.ndim < 1:
        raise ValueError("s0 must have batch dimension [B,...]")

    B = s0.shape[0]
    device = s0.device
    dtype = s0.dtype

    # init tangent
    if v0 is None:
        v = torch.randn_like(s0)
    else:
        if v0.shape != s0.shape:
            raise ValueError(f"v0 must match s0 shape, got {v0.shape} vs {s0.shape}")
        v = v0.to(device=device, dtype=dtype)

    v, _ = _normalize(v, eps=eps)

    s = s0.detach()
    total_steps = int(burn_in) + int(T)

    log_stretch = torch.zeros((total_steps, B), device=device, dtype=dtype)

    # We need autograd for JVP, so we cannot keep @torch.no_grad inside the loop for JVP.
    # We'll manually control grad.
    for t in range(total_steps):
        # enable grad on s for JVP
        s_req = s.detach().requires_grad_(True)

        # one-step forward
        s_next = F(s_req)

        # tangent pushforward: v_next = J_t v
        v_next = jvp_F(F, s_req, v)

        # record stretching factor (per batch)
        v, n = _normalize(v_next, eps=eps)
        log_stretch[t] = torch.log(n)

        # advance state (detach to avoid graph growth)
        s = s_next.detach()

    # discard burn-in
    window = log_stretch[burn_in:, :]  # [T, B]
    lambda_hat = window.mean(dim=0)    # [B]
    lambda_mean = lambda_hat.mean()    # scalar

    out: Dict[str, torch.Tensor] = {
        "lambda_hat": lambda_hat.detach().cpu(),
        "lambda_mean": lambda_mean.detach().cpu(),
    }
    if return_traj:
        out["final_state"] = s.detach().cpu()
        out["log_stretch"] = log_stretch.detach().cpu()
    return out