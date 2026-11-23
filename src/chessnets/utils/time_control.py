"""Small, model-agnostic helpers for chess time-control context."""

from __future__ import annotations

import dataclasses
import enum
import math
import typing as tp

if tp.TYPE_CHECKING:
    from chessnets.utils.board_history import BoardHistory


@dataclasses.dataclass(frozen=True)
class TimeControlMetadata:
    """Parsed initial time and increment, in seconds."""

    time_delay: int
    time_increment: int
    time_control: str


class TimeInfoPromptStyle(str, enum.Enum):
    """Wording used for the per-player clock information."""

    LEGACY = "legacy"
    REWRITTEN = "rewritten"


def time_control_to_time_control_name(time_control: str) -> str:
    """Return the conventional category for a ``base_seconds+increment`` control."""
    if "+" not in str(time_control):
        time_control = f"{time_control}+0"
    try:
        base, increment = time_control.split("+", 1)
        time_delay = int(base)
        time_increment = int(increment)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid time control: {time_control!r}") from exc

    if time_delay < 0 or time_increment < 0:
        raise ValueError(f"Time control values must be non-negative: {time_control!r}")

    estimated_game_duration = time_delay + time_increment * 40
    if estimated_game_duration < 30:
        return "UltraBullet"
    if estimated_game_duration < 180:
        return "Bullet"
    if estimated_game_duration < 480:
        return "Blitz"
    if estimated_game_duration < 1500:
        return "Rapid"
    return "Classical"


def parse_time_control(
    time_control: str | int | float | None,
) -> TimeControlMetadata | None:
    """Parse a Lichess time-control value.

    ``None``, empty strings, ``-`` and ``?`` represent missing metadata. A
    value without ``+`` is interpreted as initial seconds with no increment.
    """
    if time_control is None:
        return None
    if isinstance(time_control, float) and math.isnan(time_control):
        return None

    value = str(time_control).strip()
    if value in {"", "-", "?"}:
        return None
    if "+" not in value:
        value = f"{value}+0"

    base, increment = value.split("+", 1)
    try:
        time_delay = int(base)
        time_increment = int(increment)
    except ValueError as exc:
        raise ValueError(f"Invalid time control: {time_control!r}") from exc
    if time_delay < 0 or time_increment < 0:
        raise ValueError(f"Time control values must be non-negative: {time_control!r}")

    normalized = f"{time_delay}+{time_increment}"
    return TimeControlMetadata(
        time_delay=time_delay,
        time_increment=time_increment,
        time_control=time_control_to_time_control_name(normalized),
    )


def time_control_to_time_control_metadata(
    time_control: str | int | float | None,
) -> dict[str, tp.Any] | None:
    """Backward-compatible dictionary representation for prompt builders."""
    metadata = parse_time_control(time_control)
    return dataclasses.asdict(metadata) if metadata is not None else None


def _format_timespan(seconds: float | int | None) -> str:
    if seconds is None:
        return "N/A"
    seconds = max(0.0, float(seconds))
    parts: list[str] = []
    for unit_seconds, unit in ((3600, "hour"), (60, "minute")):
        value, seconds = divmod(seconds, unit_seconds)
        if value:
            parts.append(f"{int(value)} {unit}{'' if value == 1 else 's'}")

    if seconds or not parts:
        value = int(seconds) if seconds.is_integer() else round(seconds, 2)
        parts.append(f"{value} second{'' if value == 1 else 's'}")

    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _format_time_delay(seconds: int) -> str:
    parts = []
    for unit_seconds, unit in ((3600, "hour"), (60, "minute"), (1, "second")):
        value, seconds = divmod(seconds, unit_seconds)
        if value:
            parts.append(f"{value} {unit}{'' if value == 1 else 's'}")
    if len(parts) < 2:
        return "".join(parts)
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def append_basic_time_info_to_prompt(
    history: BoardHistory,
    index: int,
    metadata: dict[str, tp.Any] | TimeControlMetadata | None,
    prompt: str,
    time_info_prompt_style: TimeInfoPromptStyle | str = TimeInfoPromptStyle.REWRITTEN,
) -> str:
    """Append time-control and available clock context to a prompt."""
    try:
        time_info_prompt_style = TimeInfoPromptStyle(time_info_prompt_style)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported time-info prompt style: {time_info_prompt_style!r}"
        ) from exc
    if metadata is not None:
        if isinstance(metadata, TimeControlMetadata):
            metadata = dataclasses.asdict(metadata)
        time_delay = int(metadata["time_delay"])
        time_increment = int(metadata["time_increment"])
        prompt += (
            f" The time control for this {metadata['time_control'].lower()} chess game "
            f"was {_format_time_delay(time_delay)} "
            f"{'with no additional increment' if time_increment == 0 else f'with {_format_timespan(time_increment)} of increment'}."
        )

    time_info = history.get_time_info(index)
    if time_info is None:
        return prompt

    active_spent = _format_timespan(time_info.active_player_last_move_time_spent)
    opponent_spent = _format_timespan(time_info.opponent_last_move_time_spent)
    active_remaining = _format_timespan(time_info.active_player_time_remaining)
    opponent_remaining = _format_timespan(time_info.opponent_time_remaining)
    if time_info_prompt_style == TimeInfoPromptStyle.LEGACY:
        prompt += (
            f" Time taken last turn by active player: {active_spent}. "
            f"Time taken last turn by opponent: {opponent_spent}. "
            f"Time remaining for the active player: {active_remaining}. "
            f"Time remaining for the opponent: {opponent_remaining}."
        )
    else:
        prompt += (
            " The time spent by the active player on their last move was "
            f"{active_spent}, and the opponent spent {opponent_spent}. "
            f"The active player had {active_remaining} remaining, while the "
            f"opponent had {opponent_remaining} remaining."
        )
    return prompt


__all__ = [
    "TimeControlMetadata",
    "TimeInfoPromptStyle",
    "append_basic_time_info_to_prompt",
    "parse_time_control",
    "time_control_to_time_control_metadata",
    "time_control_to_time_control_name",
]
