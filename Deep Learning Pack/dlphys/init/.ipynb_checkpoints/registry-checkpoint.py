# dlphys/init/registry.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch.nn as nn

from dlphys.config.base import ExperimentConfig
from dlphys.config.registry import Registry, RegistryError

# A separate registry for init strategies (so config/registry.py doesn't need to change)
INIT_REGISTRY = Registry("inits")


def register_init(name: str, *, overwrite: bool = False):
    """
    Decorator:
      @register_init("kaiming")
      def init_fn(cfg, model, **kwargs): ...
    """
    def deco(fn: Callable[..., Dict[str, Any]]):
        INIT_REGISTRY.register(name, fn, overwrite=overwrite)
        return fn
    return deco


def apply_init(
    cfg: ExperimentConfig,
    model: nn.Module,
    *,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Apply an init strategy between build_model and build_optimizer.

    Uses:
      cfg.extra["init_name"]   (optional)
      cfg.extra["init_kwargs"] (optional dict)

    If init_name is missing, it's a no-op.

    Returns a small report dict for bookkeeping.
    """
    name = cfg.extra.get("init_name", None)
    if not name:
        return {"init_applied": False, "init_name": None}

    kwargs = dict(cfg.extra.get("init_kwargs", {}) or {})
    fn = INIT_REGISTRY.get(name)

    if logger is not None:
        logger.info(f"Applying init: {name} kwargs={kwargs}")

    report = fn(cfg, model, **kwargs) or {}
    report.setdefault("init_applied", True)
    report.setdefault("init_name", name)

    if logger is not None:
        logger.info(f"Init report: {report}")

    return report


# Ensure built-in init strategies are registered when this module is imported.
from . import standard as _standard  # noqa: E402,F401
