# SPDX-License-Identifier: MIT
# Portions adapted from Maia 2 (https://github.com/CSSLab/maia2),
# Copyright (c) 2024 CSSLab. See LICENSES/Maia2-MIT.txt.
# Modified for ChessNets by Sherman Siu; modifications are under the same license.

import logging
import pathlib
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

from chessnets._optional import import_optional
from chessnets.models.maia2.configuration_maia2 import Config
from chessnets.models.maia2.utils import (
    create_elo_dict,
    get_all_possible_moves,
)

rearrange = import_optional("einops", "maia2").rearrange


class BasicBlock(torch.nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        mid_planes = planes

        self.conv1 = torch.nn.Conv2d(
            in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(mid_planes)
        self.conv2 = torch.nn.Conv2d(
            mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = F.relu(out)

        return out


class ChessResNet(torch.nn.Module):
    def __init__(self, block: nn.Module, cfg: Config):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            cfg.input_channels,
            cfg.dim_cnn,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(cfg.dim_cnn)
        self.layers = self._make_layer(block, cfg.dim_cnn, cfg.num_blocks_cnn)
        self.conv_last = torch.nn.Conv2d(
            cfg.dim_cnn, cfg.vit_length, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_last = torch.nn.BatchNorm2d(cfg.vit_length)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, planes, stride))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.conv_last(out)
        out = self.bn_last(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EloAwareAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        elo_dim: int = 64,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.elo_query = nn.Linear(elo_dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, elo_emb: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        elo_effect = self.elo_query(elo_emb).view(x.size(0), self.heads, 1, -1)
        q = q + elo_effect

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        elo_dim: int = 64,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.elo_layers = nn.ModuleList([])
        for _ in range(depth):
            self.elo_layers.append(
                nn.ModuleList(
                    [
                        EloAwareAttention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            elo_dim=elo_dim,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, elo_emb: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.elo_layers:
            x = attn(x, elo_emb) + x
            x = ff(x) + x

        return self.norm(x)


class Maia2Model(torch.nn.Module):
    def __init__(self, output_dim: int, elo_dict: dict[str, int], cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.chess_cnn = ChessResNet(BasicBlock, cfg)

        heads = 16
        dim_head = 64
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8 * 8, cfg.dim_vit),
            nn.LayerNorm(cfg.dim_vit),
        )
        self.transformer = Transformer(
            cfg.dim_vit,
            cfg.num_blocks_vit,
            heads,
            dim_head,
            mlp_dim=cfg.dim_vit,
            dropout=0.1,
            elo_dim=cfg.elo_dim * 2,
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.vit_length, cfg.dim_vit))

        self.fc_1 = nn.Linear(cfg.dim_vit, output_dim)
        # self.fc_1_1 = nn.Linear(cfg.dim_vit, cfg.dim_vit)
        self.fc_2 = nn.Linear(cfg.dim_vit, output_dim + 6 + 6 + 1 + 64 + 64)
        # self.fc_2_1 = nn.Linear(cfg.dim_vit, cfg.dim_vit)
        self.fc_3 = nn.Linear(128, 1)
        self.fc_3_1 = nn.Linear(cfg.dim_vit, 128)

        self.elo_embedding = torch.nn.Embedding(len(elo_dict), cfg.elo_dim)

        self.dropout = nn.Dropout(p=0.1)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)

    def forward(
        self,
        boards: torch.Tensor,
        elos_self: torch.IntTensor,
        elos_oppo: torch.IntTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.cfg.input_channels, 8, 8)
        embs = self.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        x = self.to_patch_embedding(embs)
        x += self.pos_embedding
        x = self.dropout(x)

        elos_emb_self = self.elo_embedding(elos_self)
        elos_emb_oppo = self.elo_embedding(elos_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        x = self.transformer(x, elos_emb).mean(dim=1)

        x = self.last_ln(x)

        logits_maia = self.fc_1(x)
        logits_side_info = self.fc_2(x)
        logits_value = self.fc_3(self.dropout(torch.relu(self.fc_3_1(x)))).squeeze(
            dim=-1
        )

        return logits_maia, logits_side_info, logits_value

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        output_dim: int | None = None,
        elo_dict: dict[str, int] | None = None,
    ) -> tp.Self:
        all_moves = get_all_possible_moves()
        return cls(
            output_dim or len(all_moves),
            elo_dict or create_elo_dict(),
            config,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path = "shermansiu/maia2-blitz",
        device: str | torch.device = "cpu",
        *,
        time_control: tp.Literal["blitz", "rapid"] | None = None,
        variant: tp.Literal["blitz", "rapid"] | None = None,
        use_safetensors: bool | None = None,
        is_eval: bool = True,
        revision: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> tp.Self:
        aliases = {
            "blitz": ("blitz", "shermansiu/maia2-blitz"),
            "rapid": ("rapid", "shermansiu/maia2-rapid"),
            "maia2_blitz": ("blitz", "shermansiu/maia2-blitz"),
            "maia2_rapid": ("rapid", "shermansiu/maia2-rapid"),
            "maia2-blitz": ("blitz", "shermansiu/maia2-blitz"),
            "maia2-rapid": ("rapid", "shermansiu/maia2-rapid"),
        }
        if isinstance(pretrained_model_name_or_path, str):
            alias = aliases.get(pretrained_model_name_or_path)
            if alias is not None:
                selected, repository_id = alias
                if time_control is not None and time_control != selected:
                    raise ValueError(
                        "pretrained_model_name_or_path and time_control select "
                        "different Maia2 checkpoints"
                    )
                time_control = selected
                pretrained_model_name_or_path = repository_id
        if variant is not None:
            if time_control is not None and variant != time_control:
                raise ValueError(
                    "variant and time_control select different checkpoints"
                )
            time_control = variant

        resolve_kwargs = dict(
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        config = Config.from_pretrained(pretrained_model_name_or_path, **resolve_kwargs)
        configured_time_control = getattr(config, "time_control", None)
        if time_control is not None and configured_time_control != time_control:
            raise ValueError(
                f"Requested {time_control!r} but checkpoint config selects "
                f"{configured_time_control!r}"
            )
        time_control = configured_time_control
        if time_control not in {"blitz", "rapid"}:
            raise ValueError("Maia 2 config must select 'blitz' or 'rapid'")

        state_dict: dict[str, torch.Tensor]
        if use_safetensors is not False:
            from chessnets.loaders.safetensors import (
                build_with_state_dict,
                load_safetensors_file,
                resolve_safetensors_checkpoint,
            )

            try:
                model_path = resolve_safetensors_checkpoint(
                    pretrained_model_name_or_path, **resolve_kwargs
                )
            except FileNotFoundError:
                if use_safetensors:
                    raise
            else:
                state_dict = load_safetensors_file(model_path)
                model = build_with_state_dict(
                    lambda: cls.from_config(config), state_dict
                )
                model.to(device)
                if is_eval:
                    model.eval()
                logging.info(f"Model for {time_control} games loaded to {device}.")
                return model

        from chessnets.models._pretrained import resolve_pretrained_file

        model_path = resolve_pretrained_file(
            pretrained_model_name_or_path, "model.pt", **resolve_kwargs
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        model = cls.from_config(config)
        model.load_state_dict(state_dict, strict=True)

        model.to(device)

        if is_eval:
            model.eval()

        logging.info(f"Model for {time_control} games loaded to {device}.")

        return model
