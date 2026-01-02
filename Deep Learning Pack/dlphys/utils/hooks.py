# dlphys/utils/hooks.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn


Tensor = torch.Tensor


def _default_filter(name: str, module: nn.Module) -> bool:
    return True


@dataclass
class ActivationRecord:
    """
    Store activations from hooks.
    activations[name] -> list[Tensor] (each forward appends one tensor)
    """
    activations: Dict[str, List[Tensor]] = field(default_factory=dict)

    def clear(self) -> None:
        self.activations.clear()

    def add(self, name: str, x: Tensor) -> None:
        self.activations.setdefault(name, []).append(x)

    def last(self, name: str) -> Tensor:
        return self.activations[name][-1]

    def keys(self) -> List[str]:
        return list(self.activations.keys())


class HookHandleManager:
    def __init__(self) -> None:
        self._handles: List[Any] = []

    def add(self, handle: Any) -> None:
        self._handles.append(handle)

    def remove_all(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


class ActivationCacher:
    """
    Register forward hooks to cache activations.

    Example:
        ac = ActivationCacher(model, module_names=["net.0", "net.2"])
        ac.register()
        y = model(x)
        acts = ac.record.activations
        ac.remove()

    Or with context manager:
        with ActivationCacher(model, module_types=(nn.Linear,), name_filter=lambda n,m: "net" in n) as ac:
            y = model(x)
            ...
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        module_names: Optional[Sequence[str]] = None,
        module_types: Optional[Tuple[Type[nn.Module], ...]] = None,
        name_filter: Optional[Callable[[str, nn.Module], bool]] = None,
        detach: bool = True,
        to_cpu: bool = True,
        clone: bool = False,
        max_per_module: Optional[int] = None,
    ) -> None:
        self.model = model
        self.module_names = list(module_names) if module_names is not None else None
        self.module_types = module_types
        self.name_filter = name_filter or _default_filter

        self.detach = detach
        self.to_cpu = to_cpu
        self.clone = clone
        self.max_per_module = max_per_module

        self.record = ActivationRecord()
        self._mgr = HookHandleManager()
        self._registered = False

    def _select_modules(self) -> List[Tuple[str, nn.Module]]:
        named = list(self.model.named_modules())

        # If module_names given, pick exact matches
        if self.module_names is not None:
            name_to_mod = {n: m for n, m in named}
            out: List[Tuple[str, nn.Module]] = []
            for n in self.module_names:
                if n not in name_to_mod:
                    raise KeyError(
                        f"ActivationCacher: module name '{n}' not found. "
                        f"Available examples: {list(name_to_mod.keys())[:10]}"
                    )
                out.append((n, name_to_mod[n]))
            return out

        # Else select by type + filter
        out = []
        for n, m in named:
            if n == "":  # root module
                continue
            if self.module_types is not None and not isinstance(m, self.module_types):
                continue
            if not self.name_filter(n, m):
                continue
            out.append((n, m))
        return out

    def _process_tensor(self, x: Tensor) -> Tensor:
        if self.detach:
            x = x.detach()
        if self.clone:
            x = x.clone()
        if self.to_cpu:
            x = x.cpu()
        return x

    def _hook_fn(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            # Handle output being Tensor or tuple/list of Tensors
            if torch.is_tensor(output):
                y = self._process_tensor(output)
                self.record.add(name, y)
            elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
                y = self._process_tensor(output[0])
                self.record.add(name, y)
            else:
                # Ignore non-tensor outputs
                return

            # Trim history if max_per_module set
            if self.max_per_module is not None:
                lst = self.record.activations.get(name, [])
                if len(lst) > self.max_per_module:
                    self.record.activations[name] = lst[-self.max_per_module:]

        return hook

    def register(self) -> None:
        if self._registered:
            return
        modules = self._select_modules()
        for name, mod in modules:
            h = mod.register_forward_hook(self._hook_fn(name))
            self._mgr.add(h)
        self._registered = True

    def remove(self) -> None:
        self._mgr.remove_all()
        self._registered = False

    def clear(self) -> None:
        self.record.clear()

    def __enter__(self) -> "ActivationCacher":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()
        return None
