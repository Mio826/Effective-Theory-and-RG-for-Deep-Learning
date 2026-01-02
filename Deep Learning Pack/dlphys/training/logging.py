# dlphys/training/logging.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import time

from dlphys.training.callbacks import Callback, TrainState
from dlphys.utils.io import append_jsonl, ensure_dir

PathLike = Union[str, Path]


def _to_float_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        try:
            out[k] = float(v)
        except Exception:
            # skip non-numeric metrics
            pass
    return out


@dataclass
class JSONLLoggerCallback(Callback):
    """
    Write training metrics to a JSONL file.

    Each line: {"t": ..., "epoch":..., "step":..., "metrics": {...}}
    """
    path: PathLike
    every_n_steps: int = 1
    also_log_epoch_end: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def on_fit_start(self, trainer: Any, state: TrainState) -> None:
        ensure_dir(self.path.parent)
        # You can optionally write a header line; we keep it simple.

    def on_step_end(self, trainer: Any, state: TrainState, batch: Any, outputs: Any) -> None:
        if self.every_n_steps <= 0:
            return
        if state.step % self.every_n_steps != 0:
            return

        rec = {
            "t": time.time(),
            "epoch": int(state.epoch),
            "step": int(state.step),
            "metrics": _to_float_dict(state.metrics),
        }
        append_jsonl(self.path, rec)

    def on_epoch_end(self, trainer: Any, state: TrainState) -> None:
        if not self.also_log_epoch_end:
            return
        rec = {
            "t": time.time(),
            "epoch": int(state.epoch),
            "step": int(state.step),
            "metrics": _to_float_dict(state.metrics),
            "event": "epoch_end",
        }
        append_jsonl(self.path, rec)
