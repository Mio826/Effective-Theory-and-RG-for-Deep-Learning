# dlphys/config/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, TypeVar

import torch
import torch.nn as nn

from dlphys.config.base import ExperimentConfig


T = TypeVar("T")


class RegistryError(RuntimeError):
    pass


@dataclass
class Registry:
    """A minimal string -> callable registry."""
    name: str
    _items: Dict[str, Callable[..., Any]]

    def __init__(self, name: str) -> None:
        self.name = name
        self._items = {}

    def register(self, key: str, fn: Callable[..., Any], *, overwrite: bool = False) -> None:
        k = str(key)
        if (not overwrite) and (k in self._items):
            raise RegistryError(f"[{self.name}] key already registered: '{k}'")
        self._items[k] = fn

    def get(self, key: str) -> Callable[..., Any]:
        k = str(key)
        if k not in self._items:
            raise RegistryError(
                f"[{self.name}] unknown key: '{k}'. Available: {sorted(self._items.keys())}"
            )
        return self._items[k]

    def has(self, key: str) -> bool:
        return str(key) in self._items

    def keys(self) -> Tuple[str, ...]:
        return tuple(sorted(self._items.keys()))

    def clear(self) -> None:
        self._items.clear()


# Global registries (v0)
MODEL_REGISTRY = Registry("models")
OPTIM_REGISTRY = Registry("optimizers")
DATA_REGISTRY = Registry("data")  # placeholder for later


# ---- Register helpers (decorator-friendly) ----
def register_model(name: str, *, overwrite: bool = False):
    def deco(fn: Callable[..., nn.Module]):
        MODEL_REGISTRY.register(name, fn, overwrite=overwrite)
        return fn
    return deco


def register_optimizer(name: str, *, overwrite: bool = False):
    def deco(fn: Callable[..., torch.optim.Optimizer]):
        OPTIM_REGISTRY.register(name, fn, overwrite=overwrite)
        return fn
    return deco


def register_data(name: str, *, overwrite: bool = False):
    def deco(fn: Callable[..., Any]):
        DATA_REGISTRY.register(name, fn, overwrite=overwrite)
        return fn
    return deco


# ---- Build functions (use cfg.extra fields to stay v0-minimal) ----
def build_model(cfg: ExperimentConfig) -> nn.Module:
    """
    Build model using:
      cfg.extra["model_name"] (required)
      cfg.extra["model_kwargs"] (optional dict)
    """
    name = cfg.extra.get("model_name", None)
    if name is None:
        raise RegistryError("cfg.extra['model_name'] is required to build a model.")
    kwargs = cfg.extra.get("model_kwargs", {}) or {}
    ctor = MODEL_REGISTRY.get(name)
    return ctor(cfg, **kwargs)


def build_optimizer(cfg: ExperimentConfig, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """
    Build optimizer using:
      cfg.extra["optim_name"] (optional, default 'adamw')
      cfg.extra["optim_kwargs"] (optional dict)
    Falls back to cfg.lr / cfg.weight_decay when kwargs don't specify them.
    """
    name = cfg.extra.get("optim_name", "adamw")
    kwargs = dict(cfg.extra.get("optim_kwargs", {}) or {})
    # Provide defaults from cfg if user didn't set them
    kwargs.setdefault("lr", cfg.lr)
    kwargs.setdefault("weight_decay", cfg.weight_decay)

    ctor = OPTIM_REGISTRY.get(name)
    return ctor(cfg, params, **kwargs)


def build_data(cfg: ExperimentConfig) -> Any:
    """
    Placeholder for later:
      cfg.extra["data_name"] + cfg.extra["data_kwargs"]
    In v0 we just provide the hook.
    """
    name = cfg.extra.get("data_name", None)
    if name is None:
        raise RegistryError("cfg.extra['data_name'] is required to build data.")
    kwargs = cfg.extra.get("data_kwargs", {}) or {}
    ctor = DATA_REGISTRY.get(name)
    return ctor(cfg, **kwargs)


# ---- Default registrations (v0 baseline) ----
@register_optimizer("sgd", overwrite=True)
def _opt_sgd(cfg: ExperimentConfig, params: Iterable[torch.nn.Parameter], **kwargs: Any):
    return torch.optim.SGD(params, **kwargs)


@register_optimizer("adam", overwrite=True)
def _opt_adam(cfg: ExperimentConfig, params: Iterable[torch.nn.Parameter], **kwargs: Any):
    return torch.optim.Adam(params, **kwargs)


@register_optimizer("adamw", overwrite=True)
def _opt_adamw(cfg: ExperimentConfig, params: Iterable[torch.nn.Parameter], **kwargs: Any):
    return torch.optim.AdamW(params, **kwargs)




# from .toy_attention import ToyAttentionConfig, ToyAttentionDynamics

# @register_model("toy_attention", overwrite=True)
# def build_toy_attention(cfg: ExperimentConfig, **kwargs: Any) -> nn.Module:
#     # required: d_model, d_k
#     if "d_model" not in kwargs or "d_k" not in kwargs:
#         raise ValueError("toy_attention requires model_kwargs: {d_model: int, d_k: int, ...}")

#     mc = ToyAttentionConfig(
#         d_model=int(kwargs["d_model"]),
#         d_k=int(kwargs["d_k"]),
#         L=int(kwargs.get("L", 8)),
#         num_heads=int(kwargs.get("num_heads", 1)),
#         gamma=float(kwargs.get("gamma", 0.5)),
#         phi=str(kwargs.get("phi", "identity")),
#         bias=bool(kwargs.get("bias", False)),
#     )
#     return ToyAttentionDynamics(mc)