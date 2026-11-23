"""Architecture-dispatching Lc0 model loader."""

from __future__ import annotations

import pathlib
import typing as tp

from .lc0_cf.modeling_lc0_cf import BTLeelaZeroNet
from .lc0_senet.configuration_lc0_senet import LeelaNetConfig, LeelaPolicy
from .lc0_senet.modeling_lc0_senet import SeLeelaZeroNet


class AutoLc0Model:
    """Load the concrete Lc0 implementation selected by a checkpoint config."""

    @classmethod
    def from_config(cls, config: LeelaNetConfig, **kwargs: tp.Any):
        if config.model.policy == LeelaPolicy.attention:
            return BTLeelaZeroNet.from_config(config, **kwargs)
        if config.model.policy == LeelaPolicy.convolution:
            return SeLeelaZeroNet.from_config(config, **kwargs)
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
