"""Utilities for running chessnets pipelines as live chess players."""

from .adapters import PLAY_ADAPTERS, PipelineAdapterError, PipelinePlayAdapter
from .config import (
    ChallengePolicy,
    LichessBotConfig,
    MetadataConfig,
    RuntimeConfig,
    UciEngineConfig,
)
from .context import LiveGameState, MoveDecision
from .metadata import (
    FrozenMetadataResolver,
    GameAwareMetadataResolver,
    MetadataResolutionError,
    ResolvedMetadata,
)

_UCI_EXPORTS = {
    "PipelineUciEngine",
    "UciConfig",
    "UciEngine",
    "UciGoParameters",
    "UciPosition",
    "UciProtocol",
    "UciProtocolError",
    "UciServer",
    "parse_go",
    "parse_position",
    "run_uci",
}
_LICHESS_BOT_EXPORTS = {"LichessBotRunner", "LichessClient"}


def __getattr__(name: str):
    if name in _UCI_EXPORTS:
        from . import uci

        return getattr(uci, name)
    if name in _LICHESS_BOT_EXPORTS:
        from . import lichess_bot

        return getattr(lichess_bot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChallengePolicy",
    "FrozenMetadataResolver",
    "GameAwareMetadataResolver",
    "LichessBotConfig",
    "LiveGameState",
    "LichessBotRunner",
    "LichessClient",
    "MetadataConfig",
    "MetadataResolutionError",
    "MoveDecision",
    "PLAY_ADAPTERS",
    "PipelineAdapterError",
    "PipelinePlayAdapter",
    "ResolvedMetadata",
    "RuntimeConfig",
    "UciEngine",
    "UciEngineConfig",
    "UciConfig",
    "UciGoParameters",
    "UciPosition",
    "UciProtocol",
    "UciProtocolError",
    "UciServer",
    "PipelineUciEngine",
    "parse_go",
    "parse_position",
    "run_uci",
]
