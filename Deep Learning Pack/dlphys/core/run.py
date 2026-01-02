# dlphys/core/run.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from dlphys.config.base import ExperimentConfig
from dlphys.config.registry import build_model, build_optimizer
from dlphys.utils.device import get_device
from dlphys.utils.loggers import get_logger
from dlphys.utils.seed import set_seed
from dlphys.init.registry import apply_init


def parse_dtype(dtype: str) -> torch.dtype:
    """
    Map string dtype to torch.dtype.
    Supported: "float32", "float64", "bf16", "float16"
    """
    d = (dtype or "float32").lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if d not in mapping:
        raise ValueError(f"Unknown dtype='{dtype}'. Supported: {sorted(mapping.keys())}")
    return mapping[d]


@dataclass
class RunContext:
    """
    Runtime objects created by Run.prepare()/build().
    This is the "live" state of a run (not the static config).
    """
    cfg: ExperimentConfig
    device: torch.device
    dtype: torch.dtype
    run_dir: Path
    log_path: Path

    logger: Any  # logging.Logger, kept generic to avoid typing noise
    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None

    extra: Dict[str, Any] = field(default_factory=dict)


class Run:
    """
    Run = "turn a config into a runnable set of runtime objects".

    Responsibilities (v0):
      - resolve paths / create run_dir
      - set seed / device / dtype
      - configure logger (writing into run_dir)
      - build model + optimizer via registry
      - save config snapshot into run_dir
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.ctx: Optional[RunContext] = None

    def prepare(self, *, base_dir: Optional[Path] = None) -> RunContext:
        """
        Prepare runtime environment.
        Call this once before build().
        """
        # 1) Resolve run directory + log path
        self.cfg.resolve_paths(base_dir=base_dir, mkdir=True)
        run_dir = self.cfg.run_dir_abs
        log_path = self.cfg.log_path_abs

        # 2) Seed
        set_seed(self.cfg.seed, deterministic=self.cfg.deterministic)

        # 3) Device + dtype
        device = get_device(self.cfg.device)
        dtype = parse_dtype(self.cfg.dtype)

        # 4) Logger (reset=True is notebook-friendly)
        logger = get_logger(
            name=f"{self.cfg.project_name}.run",
            log_file=log_path,
            reset=True,
        )
        logger.info(f"Run prepared: run_dir={run_dir}")
        logger.info(f"device={device}, dtype={dtype}, seed={self.cfg.seed}, deterministic={self.cfg.deterministic}")

        self.ctx = RunContext(
            cfg=self.cfg,
            device=device,
            dtype=dtype,
            run_dir=run_dir,
            log_path=log_path,
            logger=logger,
        )
        return self.ctx

    def build(self) -> RunContext:
        """
        Build components using registries.
        Requires prepare() called first.
        """
        if self.ctx is None:
            raise RuntimeError("Run.build() called before prepare().")

        # Build model
        model = build_model(self.cfg)

        # Move model to device (dtype handling is optional; float32 is default)
        model = model.to(self.ctx.device)
        # If user wants float64/bf16/fp16, cast parameters/buffers
        if self.ctx.dtype is not torch.float32:
            model = model.to(self.ctx.dtype)

        # Apply init (optional): between model build and optimizer build
        init_report = apply_init(self.cfg, model, logger=self.ctx.logger)
        self.ctx.extra["init"] = init_report

        # Build optimizer
        opt = build_optimizer(self.cfg, model.parameters())


        self.ctx.model = model
        self.ctx.optimizer = opt

        self.ctx.logger.info(f"Built model: {model.__class__.__name__}")
        self.ctx.logger.info(f"Built optimizer: {opt.__class__.__name__}")
        return self.ctx

    def save_config(self, filename: str = "config.json") -> Path:
        """
        Save a snapshot of config into run_dir.
        Requires prepare() first (so run_dir exists).
        """
        if self.ctx is None:
            raise RuntimeError("Run.save_config() called before prepare().")
        path = self.ctx.run_dir / filename
        self.cfg.save(path, include_resolved=True)
        self.ctx.logger.info(f"Saved config: {path}")
        return path
