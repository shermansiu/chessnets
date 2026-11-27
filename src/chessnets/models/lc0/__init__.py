"""Lc0 and Maia-1 networks: the ChessFormer and SE-ResNet architectures.

Submodules load lazily so that importing the configuration does not import the
model implementations.
"""

from __future__ import annotations

import importlib
import typing as tp

if tp.TYPE_CHECKING:
    from .configuration_lc0 import Lc0Config, Lc0InferenceConfig, Lc0ModelConfig
    from .modeling_lc0 import AutoLc0Model
    from .modeling_lc0_chessformer import Lc0ChessFormerModel
    from .modeling_lc0_seresnet import Lc0SeResNetModel

_EXPORTS = {
    "AutoLc0Model": "modeling_lc0",
    "Lc0ChessFormerModel": "modeling_lc0_chessformer",
    "Lc0Config": "configuration_lc0",
    "Lc0InferenceConfig": "configuration_lc0",
    "Lc0ModelConfig": "configuration_lc0",
    "Lc0SeResNetModel": "modeling_lc0_seresnet",
}


def __getattr__(name: str) -> tp.Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


__all__ = [
    "AutoLc0Model",
    "Lc0ChessFormerModel",
    "Lc0Config",
    "Lc0InferenceConfig",
    "Lc0ModelConfig",
    "Lc0SeResNetModel",
]
