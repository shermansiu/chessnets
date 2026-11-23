"""Adapters from live game context to existing pipeline APIs."""

from __future__ import annotations

import asyncio
import inspect
import typing as tp

import chess

from ..pipelines.base import Pipeline
from .context import LiveGameState, MoveDecision
from .metadata import ResolvedMetadata


class PipelineAdapterError(ValueError):
    """Raised when a pipeline cannot be used for live play."""


class PipelinePlayAdapter:
    """Compatibility layer for the batch-oriented pipeline APIs."""

    def required_static_inputs(self, pipeline: Pipeline) -> set[str]:
        required = set(pipeline.input_config.required_inputs)
        return required - {"pgns", "fens"}

    def validate_launch(
        self, pipeline: Pipeline, metadata: ResolvedMetadata | None
    ) -> None:
        required = self.required_static_inputs(pipeline)
        if metadata is None:
            return
        missing = sorted(name for name in required if metadata.values.get(name) is None)
        if missing:
            raise PipelineAdapterError(
                f"Pipeline {type(pipeline).__name__} requires unresolved input(s): "
                + ", ".join(missing)
            )

    async def predict(
        self,
        pipeline: Pipeline,
        game: LiveGameState,
        metadata: ResolvedMetadata,
    ) -> MoveDecision:
        kwargs = self._build_kwargs(pipeline, game, metadata)
        result = await asyncio.to_thread(pipeline.make_prediction, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return _extract_move(result, game.board)

    def _build_kwargs(
        self,
        pipeline: Pipeline,
        game: LiveGameState,
        metadata: ResolvedMetadata,
    ) -> dict[str, tp.Any]:
        values = dict(metadata.values)
        required = set(pipeline.input_config.required_inputs)
        accepted = pipeline.input_config.allowed_inputs
        if "pgns" in required or accepted is None or "pgns" in accepted:
            values.setdefault("pgns", game.to_pgn())
        if "fens" in required or accepted is None or "fens" in accepted:
            values.setdefault("fens", game.fen)
        if "time_controls" in required:
            values.setdefault(
                "time_controls",
                (
                    metadata.model_time_control.value
                    if metadata.model_time_control
                    else None
                ),
            )
        if "prompts" in required and values.get("prompts") is None:
            raise PipelineAdapterError(
                f"Pipeline {type(pipeline).__name__} requires a prompt"
            )
        if accepted is not None:
            values = {name: value for name, value in values.items() if name in accepted}
        pipeline.input_config.validate(**values)
        kwargs = _signature_kwargs(pipeline.make_prediction, values)
        _add_if_accepted(kwargs, pipeline.make_prediction, "batch_size", 1)
        _add_if_accepted(kwargs, pipeline.make_prediction, "num_workers", 0)
        _add_if_accepted(kwargs, pipeline.make_prediction, "tqdm_leave", False)
        _add_if_accepted(kwargs, pipeline.make_prediction, "return_histories", False)
        return kwargs


class PipelineAdapterRegistry:
    def __init__(self):
        self._adapters: dict[type, PipelinePlayAdapter] = {}

    def register(self, pipeline_type: type, adapter: PipelinePlayAdapter | None = None):
        adapter = adapter or PipelinePlayAdapter()
        self._adapters[pipeline_type] = adapter
        return adapter

    def get(self, pipeline: Pipeline) -> PipelinePlayAdapter:
        for pipeline_type in type(pipeline).mro():
            if pipeline_type in self._adapters:
                return self._adapters[pipeline_type]
        return PipelinePlayAdapter()


PLAY_ADAPTERS = PipelineAdapterRegistry()


def _signature_kwargs(
    callable_: tp.Callable[..., tp.Any], values: dict[str, tp.Any]
) -> dict[str, tp.Any]:
    signature = inspect.signature(callable_)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    aliases = {
        "pgns": ("pgns", "pgn"),
        "prompts": ("prompts", "prompt"),
        "time_controls": ("time_controls", "time_control"),
        "white_elos": ("white_elos", "white_elo"),
        "black_elos": ("black_elos", "black_elo"),
        "elo_opponent": ("elo_opponent", "elo_oppo"),
        "white_clock_times": ("white_clock_times",),
        "black_clock_times": ("black_clock_times",),
        "moves_seconds": ("moves_seconds",),
        "fens": ("fens",),
        "elo_self": ("elo_self",),
    }
    result: dict[str, tp.Any] = {}
    for canonical_name, value in values.items():
        names = aliases.get(canonical_name, (canonical_name,))
        target = next((name for name in names if name in parameters), None)
        if target is not None:
            result[target] = value
        elif accepts_kwargs:
            result[canonical_name] = value
    return result


def _add_if_accepted(
    kwargs: dict[str, tp.Any],
    callable_: tp.Callable[..., tp.Any],
    name: str,
    value: tp.Any,
) -> None:
    signature = inspect.signature(callable_)
    if name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs.setdefault(name, value)


def _extract_move(result: tp.Any, board: chess.Board) -> MoveDecision:
    predictions = getattr(result, "predictions", result)
    if isinstance(predictions, (list, tuple)):
        if not predictions:
            raise PipelineAdapterError("Pipeline returned no predictions")
        prediction = predictions[0]
    else:
        prediction = predictions
    move_uci = getattr(prediction, "best_move_uci", None)
    if not move_uci:
        raise PipelineAdapterError(
            f"Pipeline returned {type(prediction).__name__} without best_move_uci"
        )
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError as exc:
        raise PipelineAdapterError(
            f"Pipeline returned invalid UCI move: {move_uci}"
        ) from exc
    if move not in board.legal_moves:
        raise PipelineAdapterError(f"Pipeline returned illegal move: {move_uci}")
    probabilities = getattr(prediction, "preds_list", None)
    if probabilities is not None and not isinstance(probabilities, dict):
        probabilities = None
    return MoveDecision(move_uci=move_uci, probabilities=probabilities)


__all__ = [
    "PLAY_ADAPTERS",
    "PipelineAdapterError",
    "PipelineAdapterRegistry",
    "PipelinePlayAdapter",
]
