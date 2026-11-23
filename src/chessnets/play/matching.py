"""Time-control normalization and matching for live pipeline inputs."""

from __future__ import annotations

import dataclasses
import typing as tp

from chessnets.utils.time_control import time_control_to_time_control_name


@dataclasses.dataclass(frozen=True, order=True)
class ExactTimeControl:
    initial_seconds: int
    increment_seconds: int

    @property
    def value(self) -> str:
        return f"{self.initial_seconds}+{self.increment_seconds}"

    @property
    def effective_seconds(self) -> int:
        return self.initial_seconds + 40 * self.increment_seconds

    @property
    def category(self) -> str:
        return time_control_to_time_control_name(self.value)


def parse_exact_time_control(value: str | int | float) -> ExactTimeControl:
    if isinstance(value, bool):
        raise ValueError(f"Invalid time control: {value!r}")
    text = str(value).strip()
    if "+" not in text:
        text += "+0"
    pieces = text.split("+")
    if len(pieces) != 2:
        raise ValueError(f"Invalid time control: {value!r}")
    try:
        initial, increment = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"Invalid time control: {value!r}") from exc
    if initial < 0 or increment < 0:
        raise ValueError(f"Time control values must be non-negative: {value!r}")
    return ExactTimeControl(initial, increment)


def time_control_from_clock(initial_ms: int, increment_ms: int) -> ExactTimeControl:
    if initial_ms < 0 or increment_ms < 0:
        raise ValueError("Lichess clock values must be non-negative")
    if initial_ms % 1000 or increment_ms % 1000:
        raise ValueError(
            "Lichess clock values are not whole seconds; this pipeline requires "
            f"seconds, got {initial_ms}ms+{increment_ms}ms"
        )
    return ExactTimeControl(initial_ms // 1000, increment_ms // 1000)


def match_time_control(
    actual: ExactTimeControl,
    supported: tp.Iterable[str],
    mode: tp.Literal["exact", "category", "closest"] = "closest",
) -> ExactTimeControl:
    candidates = tuple(parse_exact_time_control(value) for value in supported)
    if not candidates:
        raise ValueError("No supported time controls were supplied")
    exact = next((candidate for candidate in candidates if candidate == actual), None)
    if exact is not None:
        return exact
    if mode == "exact":
        raise ValueError(f"Unsupported exact time control: {actual.value}")
    if mode == "category":
        same_category = [
            candidate
            for candidate in candidates
            if candidate.category == actual.category
        ]
        if not same_category:
            raise ValueError(
                f"No supported {actual.category} time control for {actual.value}"
            )
        candidates = same_category
    return min(
        candidates,
        key=lambda candidate: (
            abs(candidate.effective_seconds - actual.effective_seconds),
            abs(candidate.initial_seconds - actual.initial_seconds),
            abs(candidate.increment_seconds - actual.increment_seconds),
        ),
    )


__all__ = [
    "ExactTimeControl",
    "match_time_control",
    "parse_exact_time_control",
    "time_control_from_clock",
]
