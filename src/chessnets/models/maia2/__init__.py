"""Maia2 model components."""

from __future__ import annotations

import typing as tp

from .configuration_maia2 import Maia2Config

__all__ = ["Maia2Config", "Maia2Model"]


def __getattr__(name: str) -> tp.Any:
    if name == "Maia2Model":
        from .modeling_maia2 import Maia2Model

        return Maia2Model
    raise AttributeError(name)
