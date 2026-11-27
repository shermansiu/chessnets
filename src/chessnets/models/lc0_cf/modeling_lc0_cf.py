"""Attention-based Lc0 model components.

Original ChessNets implementation. The architecture was derived by exporting Lc0
checkpoints to ONNX, inspecting the graph, and validating outputs numerically
against ONNX Runtime. No lczero-training source code is included.
"""

import typing as tp
from collections import OrderedDict

import attrs
import torch
from torch import nn

from chessnets.models.lc0_cf import layers_lc0_cf as layers
from chessnets.models.lc0_senet import configuration_lc0_senet as leela_config


@attrs.frozen
class MultiValueModelOutput:
    policy: torch.Tensor
    policy_optimistic_st: torch.Tensor | None
    policy_soft: torch.Tensor | None
    policy_opponent: torch.Tensor | None
    policy_next: torch.Tensor | None
    value_winner: torch.Tensor
    value_q: torch.Tensor | None
    value_q_err: torch.Tensor | None
    value_q_cat: torch.Tensor | None
    value_st: torch.Tensor | None
    value_st_err: torch.Tensor | None
    value_st_cat: torch.Tensor | None
    moves_left: torch.Tensor | None
    future: torch.Tensor | None
    outcome: torch.Tensor | None = None
    move_delay: torch.Tensor | None = None


def optional_apply(func: tp.Callable | None, x: torch.Tensor):
    if func is None:
        return None
    return func(x)


def optional_apply_triple(func: tp.Callable | None, x: torch.Tensor):
    if func is None:
        return None, None, None
    return func(x)


class BTLeelaZeroNet(nn.Module):
    def __init__(
        self,
        config: leela_config.LeelaNetConfig,
    ):
        super().__init__()
        deep_norm = layers.DeepNorm(config.model.encoder_layers)
        if config.model.arc_encoding:
            self.input_block = layers.AttentionArcEncodingInputBlock(config)
        else:
            self.input_block = layers.AttentionInputBlock(config, deep_norm)
        encoder_layers = OrderedDict(
            [
                (f"encoder_layer_{i}", layers.EncoderLayer(config, deep_norm))
                for i in range(config.model.encoder_layers)
            ]
        )
        self.encoder_layers = nn.Sequential(encoder_layers)

        self.policy_embed = nn.Linear(
            config.model.embedding_size, config.model.policy_embedding_size
        )
        self.policy_vanilla = layers.AttentionPolicyHead(config)

        if config.model.policy_optimistic_st:
            self.policy_optimistic_st = layers.AttentionPolicyHead(config)
        else:
            self.register_parameter("policy_optimistic_st", None)

        if config.model.soft_policy:
            self.policy_soft = layers.AttentionPolicyHead(
                config, depth=config.model.policy_d_aux
            )
        else:
            self.register_parameter("policy_soft", None)

        if config.model.policy_opponent:
            self.policy_opponent = layers.AttentionPolicyHead(
                config, depth=config.model.policy_d_aux, opponent=True
            )
        else:
            self.register_parameter("policy_opponent", None)

        if config.model.policy_next:
            self.policy_next = layers.AttentionPolicyHead(
                config, depth=config.model.policy_d_aux, opponent=False
            )
        else:
            self.register_parameter("policy_next", None)

        # The value head has 3 dimensions for estimating the likelihood of win/draw/loss (WDL)
        self.wdl = config.model.value.is_wdl
        self.value_winner = layers.AttentionValueHead(
            config, wdl=self.wdl, use_err=False, use_cat=False
        )

        if config.model.value_q:
            self.value_q = layers.AttentionValueHead(
                config,
                wdl=self.wdl,
                use_err=True,
                use_cat=config.model.value_q_cat,
            )
        else:
            self.register_parameter("value_q", None)

        if config.model.value_st:
            self.value_st = layers.AttentionValueHead(
                config,
                wdl=config.model.value_st_type.is_wdl,
                use_err=True,
                use_cat=config.model.value_st_cat,
            )
        else:
            self.register_parameter("value_st", None)

        if config.model.moves_left.is_enabled:
            self.moves_left_head = layers.AttentionMovesLeftHead(config)
        else:
            self.register_parameter("moves_left_head", None)

        # Future head not implemented because not enabled in BT3

        self.default_activation_fn = config.model.default_activation_fn
        self.config = config

        if config.model.policy is not leela_config.LeelaPolicy.attention:
            raise ValueError("BTLeelaZeroNet requires an attention policy config")
        if config.model.encoder_layers <= 0:
            raise ValueError("BTLeelaZeroNet requires at least one encoder layer")

    @classmethod
    def from_config(cls, config: leela_config.LeelaNetConfig) -> "BTLeelaZeroNet":
        """Initialize an attention Lc0 network from architecture config."""
        return cls(config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from chessnets.loaders.load_lc0_model import load_pretrained_model

        model = load_pretrained_model(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, cls):
            raise TypeError(
                f"Checkpoint resolved to {type(model).__name__}, not {cls.__name__}"
            )
        return model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x: torch.Tensor) -> MultiValueModelOutput:
        x = self.input_block(x)
        x = self.encoder_layers(x)
        pol_embed = self.default_activation_fn(self.policy_embed(x))

        policy = self.policy_vanilla(pol_embed)
        policy_optimistic_st = optional_apply(self.policy_optimistic_st, pol_embed)
        policy_soft = optional_apply(self.policy_soft, pol_embed)
        policy_opponent = optional_apply(self.policy_opponent, pol_embed)
        policy_next = optional_apply(self.policy_next, pol_embed)

        value_winner, _, _ = self.value_winner(x)
        value_q, value_q_err, value_q_cat = optional_apply_triple(self.value_q, x)
        value_st, value_st_err, value_st_cat = optional_apply_triple(self.value_st, x)

        moves_left = optional_apply(self.moves_left_head, x)

        return MultiValueModelOutput(
            policy,
            policy_optimistic_st,
            policy_soft,
            policy_opponent,
            policy_next,
            value_winner,
            value_q,
            value_q_err,
            value_q_cat,
            value_st,
            value_st_err,
            value_st_cat,
            moves_left,
            future=None,
        )


__all__ = [
    "BTLeelaZeroNet",
    "MultiValueModelOutput",
    "optional_apply",
    "optional_apply_triple",
]
