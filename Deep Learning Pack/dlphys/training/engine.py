# dlphys/training/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from dlphys.utils.device import to_device
from dlphys.training.callbacks import Callback, CallbackList, TrainState


@dataclass
class BatchOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]
    preds: Optional[torch.Tensor] = None


class Trainer:
    """
    Minimal training engine (v0).

    You provide:
      - model
      - optimizer
      - train_loader (iterable of batches)
      - loss_fn (callable)
      - step_fn (optional custom step)

    Engine will:
      - move batch to device
      - forward -> loss -> backward -> optimizer step
      - update TrainState
      - call callbacks at well-defined times
    """

    def __init__(
        self,
        *,
        device: torch.device,
        callbacks: Optional[CallbackList] = None,
        grad_clip: Optional[float] = None,
    ) -> None:
        self.device = device
        self.callbacks = callbacks or CallbackList()
        self.grad_clip = grad_clip

    def default_step_fn(
        self,
        model: nn.Module,
        batch: Any,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> BatchOutput:
        """
        Assumes supervised batch: (x, y)
        """
        x, y = batch
        preds = model(x)
        loss = loss_fn(preds, y)
        metrics = {"loss": float(loss.detach().cpu().item())}
        return BatchOutput(loss=loss, metrics=metrics, preds=preds)

    def fit(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Any,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        max_epochs: int = 1,
        max_steps: Optional[int] = None,
        step_fn: Optional[Callable[[nn.Module, Any, Callable], BatchOutput]] = None,
    ) -> TrainState:
        model.train()
        step_fn = step_fn or self.default_step_fn

        state = TrainState(epoch=0, step=0, max_epochs=max_epochs, max_steps=max_steps)

        self.callbacks.on_fit_start(self, state)

        for epoch in range(max_epochs):
            state.epoch = epoch
            self.callbacks.on_epoch_start(self, state)

            for batch in train_loader:
                if state.max_steps is not None and state.step >= state.max_steps:
                    break

                # move batch to device
                batch = to_device(batch, self.device)

                self.callbacks.on_step_start(self, state, batch)

                optimizer.zero_grad(set_to_none=True)

                out = step_fn(model, batch, loss_fn)
                loss = out.loss

                loss.backward()
                self.callbacks.on_backward_end(self, state, loss)

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                optimizer.step()
                self.callbacks.on_optimizer_step_end(self, state)

                # update metrics/state
                state.metrics = dict(out.metrics)
                state.step += 1

                self.callbacks.on_step_end(self, state, batch, out)

            self.callbacks.on_epoch_end(self, state)

        self.callbacks.on_fit_end(self, state)
        return state
