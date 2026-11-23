"""Frozen and game-aware metadata resolution."""

from __future__ import annotations

import dataclasses
import string
import typing as tp

from .config import MetadataConfig
from .context import LiveGameState
from .matching import ExactTimeControl, match_time_control


class MetadataResolutionError(ValueError):
    """Raised when a pipeline-required value cannot be resolved."""


@dataclasses.dataclass(frozen=True)
class ResolvedMetadata:
    """Canonical metadata passed to a pipeline adapter."""

    values: dict[str, tp.Any]
    actual_time_control: ExactTimeControl | None = None
    model_time_control: ExactTimeControl | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))


class FrozenMetadataResolver:
    def __init__(self, config: MetadataConfig):
        if config.mode != "frozen":
            raise ValueError("FrozenMetadataResolver requires metadata.mode='frozen'")
        self.config = config

    def resolve(
        self,
        game: LiveGameState,
        *,
        supported_time_controls: tp.Iterable[str] = (),
    ) -> ResolvedMetadata:
        values = _canonical_metadata(dict(self.config.values), game)
        actual = game.time_control
        model_control = _resolve_time_control(
            values.get("time_controls"),
            actual,
            supported_time_controls,
            self.config.time_control_matching,
        )
        if model_control is not None:
            values["time_controls"] = model_control.value
        return ResolvedMetadata(values, actual, model_control)


class GameAwareMetadataResolver:
    def __init__(self, config: MetadataConfig):
        if config.mode != "game":
            raise ValueError("GameAwareMetadataResolver requires metadata.mode='game'")
        self.config = config

    def resolve(
        self,
        game: LiveGameState,
        *,
        supported_time_controls: tp.Iterable[str] = (),
    ) -> ResolvedMetadata:
        values = _canonical_metadata(
            dict(self.config.values), game, orient_players=False
        )
        fallbacks = dict(self.config.fallbacks)
        if "bot_elo" in values:
            fallbacks.setdefault("bot_elo", values["bot_elo"])
        if "opponent_elo" in values:
            fallbacks.setdefault("opponent_elo", values["opponent_elo"])
        bot_elo = _first_int(game.bot_player.rating, fallbacks, "bot_elo")
        opponent_elo = _first_int(game.opponent.rating, fallbacks, "opponent_elo")
        values.pop("bot_elo", None)
        values.pop("opponent_elo", None)
        if bot_elo is not None:
            values.setdefault("elo_self", bot_elo)
        if opponent_elo is not None:
            values.setdefault("elo_opponent", opponent_elo)
        if bot_elo is not None and opponent_elo is not None:
            values["white_elos"] = bot_elo if game.bot_color else opponent_elo
            values["black_elos"] = opponent_elo if game.bot_color else bot_elo

        actual = game.time_control
        # Game-aware mode uses the actual Lichess control. The fallback is
        # only consulted for unlimited/correspondence games or incomplete
        # challenge metadata.
        values.pop("time_controls", None)
        values.pop("time_control", None)
        requested = (
            actual.value
            if actual is not None
            else self.config.fallbacks.get("time_control")
        )
        model_control = _resolve_time_control(
            requested,
            actual,
            supported_time_controls,
            self.config.time_control_matching,
        )
        if model_control is not None:
            values["time_controls"] = model_control.value

        prompt_template = values.pop("prompt_template", None)
        if prompt_template is not None:
            values["prompts"] = _format_prompt(prompt_template, game, values)
        elif "prompt" in values:
            values["prompts"] = values.pop("prompt")

        if game.white_clock_history and game.black_clock_history:
            values.setdefault("white_clock_times", game.white_clock_history)
            values.setdefault("black_clock_times", game.black_clock_history)
            values.setdefault(
                "clock_times",
                (game.white_clock_history, game.black_clock_history),
            )
        if game.move_seconds:
            values.setdefault("move_seconds", game.move_seconds)
        return ResolvedMetadata(values, actual, model_control)


def resolver_for_config(config: MetadataConfig):
    if config.mode == "frozen":
        return FrozenMetadataResolver(config)
    return GameAwareMetadataResolver(config)


def _canonical_metadata(
    values: dict[str, tp.Any],
    game: LiveGameState,
    *,
    orient_players: bool = True,
) -> dict[str, tp.Any]:
    if "prompt" in values and "prompts" not in values:
        values["prompts"] = values.pop("prompt")
    if "time_control" in values and "time_controls" not in values:
        values["time_controls"] = values.pop("time_control")
    if "white_elo" in values and "white_elos" not in values:
        values["white_elos"] = values.pop("white_elo")
    if "black_elo" in values and "black_elos" not in values:
        values["black_elos"] = values.pop("black_elo")
    if orient_players and "bot_elo" in values and "opponent_elo" in values:
        bot_elo = int(values.pop("bot_elo"))
        opponent_elo = int(values.pop("opponent_elo"))
        values.setdefault("elo_self", bot_elo)
        values.setdefault("elo_opponent", opponent_elo)
        values.setdefault("white_elos", bot_elo if game.bot_color else opponent_elo)
        values.setdefault("black_elos", opponent_elo if game.bot_color else bot_elo)
    return values


def _first_int(value: int | None, fallbacks: dict[str, tp.Any], key: str) -> int | None:
    if value is not None:
        return int(value)
    fallback = fallbacks.get(key)
    return int(fallback) if fallback is not None else None


def _resolve_time_control(
    requested: tp.Any,
    actual: ExactTimeControl | None,
    supported: tp.Iterable[str],
    mode: tp.Literal["exact", "category", "closest"],
) -> ExactTimeControl | None:
    candidates = tuple(supported)
    if requested is not None:
        from .matching import parse_exact_time_control

        requested_control = parse_exact_time_control(requested)
        if candidates:
            return match_time_control(requested_control, candidates, mode)
        return requested_control
    if actual is not None:
        if candidates:
            return match_time_control(actual, candidates, mode)
        return actual
    return None


def _format_prompt(
    template: str, game: LiveGameState, values: dict[str, tp.Any]
) -> str:
    data = {
        "game_id": game.game_id,
        "bot_elo": game.bot_player.rating,
        "opponent_elo": game.opponent.rating,
        "white_elo": game.white.rating,
        "black_elo": game.black.rating,
        "time_control": game.time_control.value if game.time_control else None,
        "fen": game.fen,
        "pgn": game.to_pgn(),
        **values,
    }
    formatter = string.Formatter()
    fields = {field for _, field, _, _ in formatter.parse(template) if field}
    missing = sorted(field for field in fields if field not in data)
    if missing:
        raise MetadataResolutionError(
            "Prompt template references unknown field(s): " + ", ".join(missing)
        )
    return template.format_map(data)


__all__ = [
    "FrozenMetadataResolver",
    "GameAwareMetadataResolver",
    "MetadataResolutionError",
    "ResolvedMetadata",
    "resolver_for_config",
]
