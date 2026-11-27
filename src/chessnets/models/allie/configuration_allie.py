# SPDX-License-Identifier: MIT
# Portions adapted from Allie (https://github.com/ippolito-cmu/allie),
# Copyright (c) 2025 Yiming Zhang. See LICENSES/Allie-MIT.txt.
# Modified for ChessNets by Sherman Siu; modifications are under the same license.

import dataclasses
import pathlib
import typing as tp

import yaml


@dataclasses.dataclass
class TokenizerConfig:
    max_length: int = 512


@dataclasses.dataclass
class DataConfig:
    tokenizer_config: TokenizerConfig = dataclasses.field(
        default_factory=TokenizerConfig
    )


@dataclasses.dataclass
class ModelConfig:
    base_model: str = "gpt2-medium"
    use_control_token: bool = True
    use_regression_head: bool = True
    use_pretrained: bool = True


@dataclasses.dataclass
class AllieConfig:
    """Model and tokenizer settings needed to load an Allie checkpoint."""

    dtype: str = "bfloat16"
    data_config: DataConfig = dataclasses.field(default_factory=DataConfig)
    model_config: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    @classmethod
    def from_dict(cls, values: dict[str, tp.Any]) -> "AllieConfig":
        data = dict(values)
        data_config = data.get("data_config") or {}
        data["data_config"] = DataConfig(
            tokenizer_config=TokenizerConfig(
                **(data_config.get("tokenizer_config") or {})
            )
        )
        data["model_config"] = ModelConfig(**(data.get("model_config") or {}))
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "AllieConfig":
        with pathlib.Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path,
        *,
        variant: str = "medium",
        revision: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        repo_type: str | None = "dataset",
    ) -> "AllieConfig":
        from chessnets.models._pretrained import resolve_pretrained_file

        return cls.from_yaml(
            resolve_pretrained_file(
                pretrained_model_name_or_path,
                f"{variant}/config.yaml",
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                repo_type=repo_type,
            )
        )
