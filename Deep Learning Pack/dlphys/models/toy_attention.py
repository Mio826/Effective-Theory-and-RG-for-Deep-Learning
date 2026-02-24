# dlphys/models/toy_attention.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn


def _make_phi(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = (name or "identity").lower()
    if name in ("identity", "id", "none"):
        return lambda x: x
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "gelu":
        return torch.nn.functional.gelu
    raise ValueError(f"Unknown phi='{name}'. Use identity/tanh/relu/gelu.")


@dataclass
class ToyAttentionConfig:
    d_model: int
    d_k: int
    L: int = 8
    num_heads: int = 1
    gamma: float = 0.5          # logits / d_k^gamma
    phi: str = "identity"       # post-attn nonlinearity
    bias: bool = False          # keep it simple: default no bias


class ToyAttentionDynamics(nn.Module):
    """
    State s_t is a stack of past tokens:
        s_t shape: [B, L+1, d_model]
    Convention:
        s_t[:, 0]   = x_t
        s_t[:, tau] = x_{t-tau}
    Update:
        x_{t+1} = phi( A_t ),   A_t = sum_tau softmax( (q·k_tau)/d_k^gamma ) * v_tau
        s_{t+1} = [x_{t+1}, x_t, ..., x_{t-L+1}]
    """

    def __init__(self, cfg: ToyAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.phi_fn = _make_phi(cfg.phi)

        H = int(cfg.num_heads)
        d_model = int(cfg.d_model)
        d_k = int(cfg.d_k)

        # Per-head Q,K,V projections (V maps back to d_model so we can sum heads easily)
        self.Wq = nn.ModuleList([nn.Linear(d_model, d_k, bias=cfg.bias) for _ in range(H)])
        self.Wk = nn.ModuleList([nn.Linear(d_model, d_k, bias=cfg.bias) for _ in range(H)])
        self.Wv = nn.ModuleList([nn.Linear(d_model, d_model, bias=cfg.bias) for _ in range(H)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: [B, L+1, d_model]
        return s_next: [B, L+1, d_model]
        """
        if s.ndim != 3:
            raise ValueError(f"s must be [B, L+1, d_model], got shape={tuple(s.shape)}")

        B, T, d_model = s.shape
        L = int(self.cfg.L)
        if T != L + 1:
            raise ValueError(f"Expected s.shape[1]==L+1=={L+1}, got {T}")

        x_t = s[:, 0, :]          # [B, d_model]
        x_hist = s               # [B, L+1, d_model] includes x_t itself

        d_k = float(self.cfg.d_k)
        scale = d_k ** float(self.cfg.gamma)

        # Aggregate heads with 1/sqrt(H) normalization (your multihead-Gaussianization knob)
        H = int(self.cfg.num_heads)
        A = torch.zeros((B, d_model), device=s.device, dtype=s.dtype)

        for h in range(H):
            q = self.Wq[h](x_t)                 # [B, d_k]
            k = self.Wk[h](x_hist)              # [B, L+1, d_k]
            v = self.Wv[h](x_hist)              # [B, L+1, d_model]

            # logits_tau = q · k_tau
            logits = torch.einsum("bd,btd->bt", q, k) / scale     # [B, L+1]
            alpha = self.softmax(logits)                          # [B, L+1]

            A_h = torch.einsum("bt,btd->bd", alpha, v)            # [B, d_model]
            A = A + A_h

        A = A / (H ** 0.5)
        x_next = self.phi_fn(A)                                   # [B, d_model]

        # shift state: new stack = [x_{t+1}, x_t, ..., x_{t-L+1}]
        s_next = torch.cat([x_next[:, None, :], s[:, :L, :]], dim=1)
        return s_next