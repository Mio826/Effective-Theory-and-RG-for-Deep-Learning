# dlphys/core/run_dynamics.py
from __future__ import annotations
from typing import Dict, List, Optional

import torch

from dlphys.config.base import ExperimentConfig
from dlphys.config.registry import build_model


@torch.no_grad()
def rollout(
    cfg: ExperimentConfig,
    *,
    s0: torch.Tensor,
    T: int,
    device: Optional[str] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Inference-only rollout for dynamical system:
        s_{t+1} = model(s_t)
    Returns states on CPU for easy analysis.
    """
    import dlphys.models  # trigger registrations

    model = build_model(cfg)
    if device is None:
        device = getattr(cfg, "device", "cpu")
    model = model.to(device)
    model.eval()

    s = s0.to(device)
    states = [s.detach().cpu()]

    for _ in range(int(T)):
        s = model(s)
        states.append(s.detach().cpu())

    return {"states": states}