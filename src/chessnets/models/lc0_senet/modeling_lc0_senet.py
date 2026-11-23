"""Convolutional (SE-ResNet) Lc0 networks, as used by Maia-1 and older Lc0 nets.

Original ChessNets implementation; see ``layers_lc0_senet`` for provenance.
"""

from __future__ import annotations

from collections import OrderedDict
import typing as tp

import torch
from torch import nn

import chessnets.models.lc0_senet.configuration_lc0_senet as leela_config
from chessnets.models.lc0_senet.layers_lc0_senet import (
    ConvBatchNormReLU,
    ConvPolicyHead,
    ConvValueHead,
    SEResidualBlock,
)

NUM_PLANES = 112
VALUE_PROJECTION_CHANNELS = 32
VALUE_HIDDEN_SIZE = 128


class PolicyValueModelOutput(tp.NamedTuple):
    """Raw network outputs: 1858 policy logits and win/draw/loss logits."""

    policy: torch.Tensor
    value: torch.Tensor


class SeLeelaZeroNet(nn.Module):
    """SE-ResNet Lc0 network with a convolutional policy head and WDL value head."""

    def __init__(self, num_filters: int, num_residual_blocks: int, se_ratio: int):
        super().__init__()
        self.input_block = ConvBatchNormReLU(NUM_PLANES, num_filters, 3)
        self.residual_blocks = nn.Sequential(
            OrderedDict(
                (f"residual_block_{index}", SEResidualBlock(num_filters, se_ratio))
                for index in range(num_residual_blocks)
            )
        )
        self.policy_head = ConvPolicyHead(num_filters)
        self.value_head = ConvValueHead(
            num_filters, VALUE_PROJECTION_CHANNELS, VALUE_HIDDEN_SIZE, 3
        )

    @classmethod
    def from_config(
        cls,
        config: leela_config.LeelaNetConfig | leela_config.LeelaInferenceConfig,
    ):
        model = cls(
            num_filters=config.model.filters,
            num_residual_blocks=config.model.residual_blocks,
            se_ratio=config.model.se_ratio,
        )
        model.config = config
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if hasattr(pretrained_model_name_or_path, "model"):
            return cls.from_config(pretrained_model_name_or_path)
        from chessnets.loaders.load_lc0_model import load_pretrained_model

        model = load_pretrained_model(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, cls):
            raise TypeError(
                f"Checkpoint resolved to {type(model).__name__}, not {cls.__name__}"
            )
        return model

    def forward(self, input_planes: torch.Tensor) -> PolicyValueModelOutput:
        trunk = self.input_block(input_planes.reshape(-1, NUM_PLANES, 8, 8))
        trunk = self.residual_blocks(trunk)
        return PolicyValueModelOutput(
            policy=self.policy_head(trunk), value=self.value_head(trunk)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


__all__ = ["PolicyValueModelOutput", "SeLeelaZeroNet"]
