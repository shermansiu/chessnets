# SPDX-License-Identifier: MIT
# Portions adapted from Maia 2 (https://github.com/CSSLab/maia2),
# Copyright (c) 2024 CSSLab. See LICENSES/Maia2-MIT.txt.
# Modified for ChessNets by Sherman Siu; modifications are under the same license.

"""Configuration loading for Maia 2 checkpoints."""

from __future__ import annotations

import json
import pathlib
import typing as tp

import yaml


class Config:
    """A small, validated Maia 2 inference configuration."""

    model_type = "maia2"
    config_schema_version = 1
    architectures = ["Maia2Model"]
    _ARCHITECTURE_FIELDS = (
        "dim_cnn",
        "dim_vit",
        "num_blocks_cnn",
        "num_blocks_vit",
        "input_channels",
        "vit_length",
        "elo_dim",
    )

    def __init__(self, config_dict: dict[str, tp.Any]):
        values = dict(config_dict)
        model = values.pop("model", None)
        if model is not None:
            if not isinstance(model, dict):
                raise ValueError("Maia 2 config 'model' must be an object")
            values.update(model)
        for key, value in values.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, values: dict[str, tp.Any]) -> "Config":
        if not isinstance(values, dict):
            raise TypeError("Maia 2 config must be a mapping")
        config = cls(values)
        for field in cls._ARCHITECTURE_FIELDS:
            if not hasattr(config, field):
                raise ValueError(f"Maia 2 config is missing {field!r}")
        model_type = getattr(config, "model_type", cls.model_type)
        if model_type != cls.model_type:
            raise ValueError(f"Unsupported Maia 2 model_type: {model_type!r}")
        schema = getattr(config, "config_schema_version", cls.config_schema_version)
        if schema != cls.config_schema_version:
            raise ValueError(f"Unsupported Maia 2 config schema: {schema!r}")
        architectures = getattr(config, "architectures", cls.architectures)
        if architectures != cls.architectures:
            raise ValueError(f"Unsupported Maia 2 architectures: {architectures!r}")
        time_control = getattr(config, "time_control", None)
        if time_control is not None and time_control not in {"blitz", "rapid"}:
            raise ValueError("Maia 2 time_control must be 'blitz' or 'rapid'")
        return config

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "Config":
        values = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(values)

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "Config":
        """Load an upstream training config for migration only."""
        with pathlib.Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})

    def to_dict(self) -> dict[str, tp.Any]:
        model = {field: getattr(self, field) for field in self._ARCHITECTURE_FIELDS}
        for field in ("side_info", "value"):
            if hasattr(self, field):
                model[field] = getattr(self, field)
        values: dict[str, tp.Any] = {
            "architectures": list(self.architectures),
            "config_schema_version": self.config_schema_version,
            "model": model,
            "model_type": self.model_type,
        }
        if hasattr(self, "time_control"):
            values["time_control"] = self.time_control
        return values

    def save_pretrained(self, directory: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(directory) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path,
        *,
        revision: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> "Config":
        from chessnets.models._pretrained import resolve_pretrained_file

        return cls.from_json(
            resolve_pretrained_file(
                pretrained_model_name_or_path,
                "config.json",
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
        )


def parse_args(cfg_file_path):
    return Config.from_yaml(cfg_file_path)


__all__ = ["Config", "parse_args"]
