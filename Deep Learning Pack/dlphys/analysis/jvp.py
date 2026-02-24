# dlphys/analysis/jvp.py
from __future__ import annotations
from typing import Callable
import torch

def jvp_F(F: Callable[[torch.Tensor], torch.Tensor],
          s: torch.Tensor,
          v: torch.Tensor) -> torch.Tensor:
    """
    Compute J(s) v where J = dF/ds, using autograd JVP.
    This is the mathematically correct object for tangent dynamics:
        δs_{t+1} = J_t δs_t.
    """
    if s.shape != v.shape:
        raise ValueError(f"s and v must have same shape, got {s.shape} vs {v.shape}")

    # Make sure s requires grad for autograd.functional.jvp
    s_req = s.detach().requires_grad_(True)
    v_det = v.detach()

    _, jvp = torch.autograd.functional.jvp(F, (s_req,), (v_det,), create_graph=False)
    return jvp