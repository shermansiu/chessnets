"""Inference configuration for the UniMaia model family.

The loader intentionally accepts legacy training YAMLs and keeps only fields
needed to reconstruct an inference model.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
import typing as tp

import yaml

from chessnets.models.lc0_senet import configuration_lc0_senet as leela_config


class TextEncoderType(leela_config.AutoStrEnum):
    frozen = enum.auto()
    lora = enum.auto()
    full_finetune = enum.auto()


class VLMModelArchitectureType(leela_config.AutoStrEnum):
    vlm = enum.auto()
    controlnet = enum.auto()
    controlnet_controllable_policy_head = enum.auto()
    cross_encoder = enum.auto()
    cross_encoder_controllable_policy_head = enum.auto()


UNIMAIA_OUTCOME_LABELS = (
    "Resigned",
    "Time forfeit",
    "Checkmate",
    "Threefold repetition",
    "Stalemate",
    "Insufficient material",
    "Draw by agreement",
    "Fivefold repetition",
    "Fifty moves",
    "Rules infraction",
    "Abandoned",
    "Other",
)


@dataclasses.dataclass
class UniMaiaConfig:
    """Model and checkpoint settings required for inference."""

    lc0_model_id: str
    llm_model_id: str
    model_dtype: tp.Literal["float32", "float16", "bfloat16"] = "bfloat16"
    text_encoder_type: TextEncoderType = TextEncoderType.frozen
    vlm_architecture_type: VLMModelArchitectureType = VLMModelArchitectureType.vlm

    use_intermediate_states_for_controlnet: bool = True
    should_use_gated_attention: bool = False
    use_key_padding_mask: bool = False

    should_use_value_loss: bool = False
    should_use_moves_left_loss: bool = False
    should_use_outcome_type_loss: bool = False
    should_use_move_delay_loss: bool = False
    num_outcome_types: int = len(UNIMAIA_OUTCOME_LABELS)
    outcome_labels: tuple[str, ...] = UNIMAIA_OUTCOME_LABELS

    fusion_num_attn_heads: int = 16
    fusion_attn_dropout: float = 0.0
    fusion_ffn_intermediate_dim_multiplier: int = 4

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_rslora: bool = True
    use_dora: bool = True
    target_lora_modules: list[str] = dataclasses.field(
        default_factory=lambda: [
            "attention.query_key_value",
            "attention.dense",
        ]
    )

    model_path: str | None = None
    tokenizer_path: str | None = None
    llm_config_path: str | None = None
    lc0_config_path: str | None = None
    device: str | None = None
    lc0_config: dict[str, tp.Any] | None = None

    @classmethod
    def from_dict(cls, values: dict[str, tp.Any]) -> "UniMaiaConfig":
        """Build a config while ignoring legacy training-only keys."""
        if not isinstance(values, dict):
            raise TypeError(f"Expected a mapping, got {type(values).__name__}")
        values = dict(values)
        if "lc0_model_id" not in values:
            for legacy_name in ("lc0_bt_model_id", "base_lc0_model_id"):
                if legacy_name in values:
                    values["lc0_model_id"] = values[legacy_name]
                    break
        if "llm_model_id" not in values:
            for legacy_name in ("text_model_id", "text_encoder_model_id"):
                if legacy_name in values:
                    values["llm_model_id"] = values[legacy_name]
                    break
        if "model_dtype" not in values and "dtype" in values:
            values["model_dtype"] = values["dtype"]
        if "outcome_labels" not in values and "outcome_type_labels" in values:
            values["outcome_labels"] = values["outcome_type_labels"]
        missing = [
            name for name in ("lc0_model_id", "llm_model_id") if not values.get(name)
        ]
        if missing:
            raise ValueError(
                "UniMaia configuration is missing required model field(s): "
                + ", ".join(missing)
            )
        fields = {field.name for field in dataclasses.fields(cls)}
        filtered = {key: value for key, value in values.items() if key in fields}
        if "model_dtype" in filtered:
            filtered["model_dtype"] = (
                str(filtered["model_dtype"]).removeprefix("torch.").lower()
            )
        for key, enum_type in (
            ("text_encoder_type", TextEncoderType),
            ("vlm_architecture_type", VLMModelArchitectureType),
        ):
            if key in filtered and not isinstance(filtered[key], enum_type):
                try:
                    filtered[key] = enum_type(str(filtered[key]).lower())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid {key}: {filtered[key]!r}; expected one of "
                        f"{[member.value for member in enum_type]}"
                    ) from exc
        labels_were_provided = "outcome_labels" in filtered
        if labels_were_provided:
            filtered["outcome_labels"] = tuple(filtered["outcome_labels"])
        if (
            not labels_were_provided
            and "num_outcome_types" in filtered
            and filtered["num_outcome_types"] <= len(UNIMAIA_OUTCOME_LABELS)
        ):
            filtered["outcome_labels"] = UNIMAIA_OUTCOME_LABELS[
                : filtered["num_outcome_types"]
            ]
        config = cls(**filtered)
        if config.num_outcome_types <= 0:
            raise ValueError("num_outcome_types must be positive")
        if config.num_outcome_types != len(config.outcome_labels):
            raise ValueError(
                "num_outcome_types must match the number of outcome_labels "
                f"({config.num_outcome_types} != {len(config.outcome_labels)})"
            )
        return config

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "UniMaiaConfig":
        with pathlib.Path(path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        return cls.from_dict(values)

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "UniMaiaConfig":
        with pathlib.Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_dict(self) -> dict[str, tp.Any]:
        def plain(value: tp.Any) -> tp.Any:
            if isinstance(value, enum.Enum):
                return value.value
            if isinstance(value, tuple):
                return [plain(item) for item in value]
            if isinstance(value, list):
                return [plain(item) for item in value]
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            return value

        values = {
            field.name: plain(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }
        return {
            "model_type": "unimaia",
            "config_schema_version": 1,
            "architectures": ["UniMaiaModel"],
            **values,
        }

    def save_pretrained(self, directory: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(directory) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def load_config(config_path: str | pathlib.Path | None = None) -> dict[str, tp.Any]:
    """Load a YAML mapping without applying training defaults."""
    if config_path is None:
        return {}
    with pathlib.Path(config_path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


__all__ = [
    "TextEncoderType",
    "UNIMAIA_OUTCOME_LABELS",
    "UniMaiaConfig",
    "VLMModelArchitectureType",
    "load_config",
]
