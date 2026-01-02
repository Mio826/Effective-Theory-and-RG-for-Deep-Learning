# dlphys/training/callbacks.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch


@dataclass
class TrainState:
    """Minimal mutable training state shared across engine + callbacks."""
    epoch: int = 0
    step: int = 0  # global step (batches)
    max_epochs: int = 1
    max_steps: Optional[int] = None

    # latest batch metrics (engine will update this each step)
    metrics: Dict[str, float] = field(default_factory=dict)

    # allow callbacks to store stuff
    extras: Dict[str, Any] = field(default_factory=dict)

    def should_stop(self) -> bool:
        if self.max_steps is not None and self.step >= self.max_steps:
            return True
        if self.epoch >= self.max_epochs:
            return True
        return False


class Callback:
    """
    Base callback with no-op hooks.
    Override what you need.
    """
    priority: int = 0  # smaller runs earlier

    def on_fit_start(self, trainer: Any, state: TrainState) -> None: ...
    def on_fit_end(self, trainer: Any, state: TrainState) -> None: ...

    def on_epoch_start(self, trainer: Any, state: TrainState) -> None: ...
    def on_epoch_end(self, trainer: Any, state: TrainState) -> None: ...

    def on_step_start(self, trainer: Any, state: TrainState, batch: Any) -> None: ...
    def on_step_end(self, trainer: Any, state: TrainState, batch: Any, outputs: Any) -> None: ...

    def on_backward_end(self, trainer: Any, state: TrainState, loss: torch.Tensor) -> None: ...
    def on_optimizer_step_end(self, trainer: Any, state: TrainState) -> None: ...


class CallbackList(Callback):
    """A thin wrapper to manage and call multiple callbacks in order."""
    def __init__(self, callbacks: Optional[Iterable[Callback]] = None) -> None:
        self.callbacks: List[Callback] = sorted(list(callbacks or []), key=lambda c: c.priority)

    def add(self, cb: Callback) -> None:
        self.callbacks.append(cb)
        self.callbacks.sort(key=lambda c: c.priority)

    # Dispatch methods
    def on_fit_start(self, trainer: Any, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_fit_start(trainer, state)

    def on_fit_end(self, trainer: Any, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_fit_end(trainer, state)

    def on_epoch_start(self, trainer: Any, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, state)

    def on_epoch_end(self, trainer: Any, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, state)

    def on_step_start(self, trainer: Any, state: TrainState, batch: Any) -> None:
        for cb in self.callbacks:
            cb.on_step_start(trainer, state, batch)

    def on_step_end(self, trainer: Any, state: TrainState, batch: Any, outputs: Any) -> None:
        for cb in self.callbacks:
            cb.on_step_end(trainer, state, batch, outputs)

    def on_backward_end(self, trainer: Any, state: TrainState, loss: torch.Tensor) -> None:
        for cb in self.callbacks:
            cb.on_backward_end(trainer, state, loss)

    def on_optimizer_step_end(self, trainer: Any, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_optimizer_step_end(trainer, state)
