"""Configuration objects for live chess runners.

The configuration deliberately contains no token value.  Tokens should be
provided by an environment variable, a secret manager, or directly by a
caller in a notebook.
"""

from __future__ import annotations

import dataclasses
import pathlib
import typing as tp

import yaml


@dataclasses.dataclass(frozen=True)
class MetadataConfig:
    """Select how pipeline metadata is supplied for each game."""

    mode: tp.Literal["frozen", "game"] = "frozen"
    values: dict[str, tp.Any] = dataclasses.field(default_factory=dict)
    fallbacks: dict[str, tp.Any] = dataclasses.field(default_factory=dict)
    time_control_matching: tp.Literal["exact", "category", "closest"] = "closest"

    def __post_init__(self) -> None:
        if self.mode not in {"frozen", "game"}:
            raise ValueError("metadata.mode must be 'frozen' or 'game'")
        if self.time_control_matching not in {"exact", "category", "closest"}:
            raise ValueError(
                "metadata.time_control_matching must be exact, category, or closest"
            )
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(self, "fallbacks", dict(self.fallbacks))


@dataclasses.dataclass(frozen=True)
class ChallengePolicy:
    """Filters applied before accepting incoming Lichess challenges."""

    variants: tuple[str, ...] = ("standard",)
    speeds: tuple[str, ...] = ()
    rated: bool | None = False
    allow_bots: bool = True
    max_concurrent_games: int = 1
    accept_incoming: bool = True

    def __post_init__(self) -> None:
        if self.max_concurrent_games <= 0:
            raise ValueError("max_concurrent_games must be positive")
        variants = (
            (self.variants,) if isinstance(self.variants, str) else tuple(self.variants)
        )
        speeds = (self.speeds,) if isinstance(self.speeds, str) else tuple(self.speeds)
        object.__setattr__(self, "variants", variants)
        object.__setattr__(self, "speeds", speeds)


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """Operational safeguards for a live runner."""

    inference_timeout_seconds: float | None = None
    clock_safety_margin_seconds: float = 1.5
    reconnect_max_seconds: float = 60.0
    request_timeout_seconds: float = 30.0
    log_format: tp.Literal["text", "json"] = "text"

    def __post_init__(self) -> None:
        if (
            self.inference_timeout_seconds is not None
            and self.inference_timeout_seconds <= 0
        ):
            raise ValueError("inference_timeout_seconds must be positive")
        if self.clock_safety_margin_seconds < 0:
            raise ValueError("clock_safety_margin_seconds must be non-negative")
        if self.reconnect_max_seconds <= 0:
            raise ValueError("reconnect_max_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.log_format not in {"text", "json"}:
            raise ValueError("runtime.log_format must be text or json")


@dataclasses.dataclass(frozen=True)
class LichessBotConfig:
    """Complete model and runner configuration."""

    model_id: str
    device: str | None = None
    model_kwargs: dict[str, tp.Any] = dataclasses.field(default_factory=dict)
    metadata: MetadataConfig = dataclasses.field(default_factory=MetadataConfig)
    challenges: ChallengePolicy = dataclasses.field(default_factory=ChallengePolicy)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        object.__setattr__(self, "model_kwargs", dict(self.model_kwargs))

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> "LichessBotConfig":
        """Load and validate a configuration mapping.

        Unknown top-level and nested keys are rejected so a misspelled safety
        option cannot silently change bot behavior.
        """

        data = dict(data)
        _reject_unknown(data, {"model", "metadata", "challenges", "runtime"}, "root")
        model = _mapping(data.get("model"), "model")
        _reject_unknown(model, {"id", "device", "kwargs"}, "model")
        metadata = _mapping(data.get("metadata"), "metadata")
        _reject_unknown(
            metadata,
            {"mode", "values", "fallbacks", "time_control_matching"},
            "metadata",
        )
        challenges = _mapping(data.get("challenges"), "challenges")
        _reject_unknown(
            challenges,
            {
                "variants",
                "speeds",
                "rated",
                "allow_bots",
                "max_concurrent_games",
                "accept_incoming",
            },
            "challenges",
        )
        runtime = _mapping(data.get("runtime"), "runtime")
        _reject_unknown(
            runtime,
            {
                "inference_timeout_seconds",
                "clock_safety_margin_seconds",
                "reconnect_max_seconds",
                "request_timeout_seconds",
                "log_format",
            },
            "runtime",
        )
        return cls(
            model_id=_required(model, "id", "model.id"),
            device=model.get("device"),
            model_kwargs=_mapping(model.get("kwargs", {}), "model.kwargs"),
            metadata=MetadataConfig(**metadata),
            challenges=ChallengePolicy(**challenges),
            runtime=RuntimeConfig(**runtime),
        )

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "LichessBotConfig":
        path = pathlib.Path(path)
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration root must be a mapping: {path}")
        return cls.from_dict(data)


@dataclasses.dataclass(frozen=True)
class UciEngineConfig:
    """Configuration for exposing a pipeline through the UCI protocol.

    UCI does not provide reliable player ratings, prompts, or the original
    time control.  Consequently metadata is deliberately frozen for the
    lifetime of an engine process.  Keeping this restriction in the config
    object prevents a GUI from silently changing the model's input contract.
    """

    model_id: str
    device: str | None = None
    model_kwargs: dict[str, tp.Any] = dataclasses.field(default_factory=dict)
    metadata: MetadataConfig = dataclasses.field(default_factory=MetadataConfig)
    name: str = "chessnets"
    author: str = "chessnets"
    inference_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if self.metadata.mode != "frozen":
            raise ValueError("UCI engines require metadata.mode='frozen'")
        if not self.name:
            raise ValueError("engine name is required")
        if not self.author:
            raise ValueError("engine author is required")
        if any(character in self.name for character in "\r\n"):
            raise ValueError("engine name must not contain newlines")
        if any(character in self.author for character in "\r\n"):
            raise ValueError("engine author must not contain newlines")
        if (
            self.inference_timeout_seconds is not None
            and self.inference_timeout_seconds <= 0
        ):
            raise ValueError("inference_timeout_seconds must be positive")
        object.__setattr__(self, "model_kwargs", dict(self.model_kwargs))

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> "UciEngineConfig":
        """Load a strict UCI configuration mapping."""

        data = dict(data)
        _reject_unknown(data, {"model", "metadata", "engine"}, "root")
        model = _mapping(data.get("model"), "model")
        _reject_unknown(model, {"id", "device", "kwargs"}, "model")
        metadata = _mapping(data.get("metadata"), "metadata")
        _reject_unknown(
            metadata,
            {"mode", "values", "fallbacks", "time_control_matching"},
            "metadata",
        )
        engine = _mapping(data.get("engine"), "engine")
        _reject_unknown(
            engine,
            {"name", "author", "inference_timeout_seconds"},
            "engine",
        )
        return cls(
            model_id=_required(model, "id", "model.id"),
            device=model.get("device"),
            model_kwargs=_mapping(model.get("kwargs", {}), "model.kwargs"),
            metadata=MetadataConfig(**metadata),
            name=engine.get("name", "chessnets"),
            author=engine.get("author", "chessnets"),
            inference_timeout_seconds=engine.get("inference_timeout_seconds"),
        )

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "UciEngineConfig":
        path = pathlib.Path(path)
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration root must be a mapping: {path}")
        return cls.from_dict(data)


def _mapping(value: tp.Any, name: str) -> dict[str, tp.Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _required(mapping: dict[str, tp.Any], key: str, name: str) -> tp.Any:
    if key not in mapping or mapping[key] is None or mapping[key] == "":
        raise ValueError(f"{name} is required")
    return mapping[key]


def _reject_unknown(mapping: dict[str, tp.Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} configuration field(s): {', '.join(unknown)}")


UciConfig = UciEngineConfig


__all__ = [
    "ChallengePolicy",
    "LichessBotConfig",
    "MetadataConfig",
    "RuntimeConfig",
    "UciConfig",
    "UciEngineConfig",
]
