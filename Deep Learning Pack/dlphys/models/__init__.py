# dlphys/models/__init__.py
"""
Model zoo for dlphys.

Important:
  Importing this package registers model builders into dlphys.config.registry.MODEL_REGISTRY
  via dlphys.models.registry (side effect).
"""

from .mlp import MLP
from .convnet import ConvNetSmall
from .resnet import ResNetCIFAR, resnet18_cifar

# Trigger registrations (side-effect)
from . import registry as _registry  # noqa: F401

__all__ = [
    "MLP",
    "ConvNetSmall",
    "ResNetCIFAR",
    "resnet18_cifar",
]
