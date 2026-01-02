# dlphys/utils/types.py
from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch

# Common aliases used across trainer/callbacks/metrics.

Tensor = torch.Tensor
Metrics = Dict[str, float]

# A batch could be:
# - (x, y)
# - dict[str, Tensor]
# - nested tuples/lists/dicts
Batch = Any

# Optional: for configs or structured dicts
JSONDict = Dict[str, Any]
