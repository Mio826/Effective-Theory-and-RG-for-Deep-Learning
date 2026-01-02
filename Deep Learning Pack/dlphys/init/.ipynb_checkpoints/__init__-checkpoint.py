# dlphys/init/__init__.py
"""
Initialization / calibration strategies.

Importing this package registers built-in init strategies.
"""

from .registry import apply_init, register_init, INIT_REGISTRY  # noqa: F401
from . import standard as _standard  # noqa: F401  (side-effect: register builtins)

__all__ = ["apply_init", "register_init", "INIT_REGISTRY"]
