"""Architecture-dispatching Lc0 model loader."""

from __future__ import annotations

import pathlib
import typing as tp

from .configuration_lc0 import Lc0Config, Lc0Policy
from .modeling_lc0_chessformer import Lc0ChessFormerModel
from .modeling_lc0_seresnet import Lc0SeResNetModel


class AutoLc0Model:
    """Load the concrete Lc0 implementation selected by a checkpoint config."""

    @classmethod
    def from_config(cls, config: Lc0Config, **kwargs: tp.Any):
        if config.model.policy == Lc0Policy.attention:
            return Lc0ChessFormerModel.from_config(config, **kwargs)
        if config.model.policy == Lc0Policy.convolution:
            return Lc0SeResNetModel.from_config(config, **kwargs)
        raise ValueError(
            f"Unsupported Lc0 policy architecture: {config.model.policy!r}"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path,
        **kwargs: tp.Any,
    ):
        from chessnets.loaders.load_lc0_model import load_pretrained_model

        return load_pretrained_model(pretrained_model_name_or_path, **kwargs)


__all__ = ["AutoLc0Model"]
