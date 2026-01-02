# dlphys/data/__init__.py
"""
Data module for dlphys.

Importing this package registers data builders into dlphys.config.registry.DATA_REGISTRY.
"""

from .datamodule import DataLoaders
from . import registry as _registry  # side-effect: registrations

__all__ = ["DataLoaders"]
