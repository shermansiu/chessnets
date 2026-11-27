"""UniMaia model configuration and inference support."""

from __future__ import annotations

import typing as tp

from .configuration_unimaia import (
    UNIMAIA_OUTCOME_LABELS,
    TextEncoderType,
    UniMaiaConfig,
    VLMModelArchitectureType,
)

__all__ = [
    "TextEncoderType",
    "UNIMAIA_OUTCOME_LABELS",
    "UniMaiaConfig",
    "VLMModelArchitectureType",
    "UniMaiaModel",
]


def __getattr__(name: str) -> tp.Any:
    if name == "UniMaiaModel":
        from .modeling_unimaia import UniMaiaModel

        return UniMaiaModel
    raise AttributeError(name)
