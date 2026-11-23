import asyncio
import types

import chess
import pytest

from chessnets.pipelines.base import PipelineInputConfig
from chessnets.play.adapters import PipelinePlayAdapter
from chessnets.play.config import MetadataConfig, UciEngineConfig
from chessnets.play.context import MoveDecision
from chessnets.play.uci import (
    PipelineUciEngine,
    UciGoParameters,
    UciPosition,
    UciProtocol,
)


class FakePipeline:
    input_config = PipelineInputConfig.from_inputs(required=("pgns",))

    def make_prediction(self, pgns, **kwargs):
        assert "[White" in pgns
        return types.SimpleNamespace(
            predictions=[types.SimpleNamespace(best_move_uci="e2e4")]
        )


class ImmediateAdapter(PipelinePlayAdapter):
    async def predict(self, pipeline, game, metadata):
        return MoveDecision("e2e4")


def _engine():
    config = UciEngineConfig(
        model_id="fake",
        metadata=MetadataConfig(
            values={"bot_elo": 1500, "opponent_elo": 1400, "time_control": "300+3"}
        ),
    )
    return PipelineUciEngine(FakePipeline(), config, adapter=ImmediateAdapter())


def test_uci_position_supports_startpos_and_custom_fen():
    position = UciPosition.startpos(["e2e4", "e7e5"])
    assert position.board.turn == chess.WHITE
    assert position.to_pgn().count("e4") == 1

    custom = UciPosition.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1", ["e1d1"])
    assert custom.board.fen().startswith("4k3/8/8/8/8/8/8/3K4")
    assert "SetUp" in custom.to_pgn()


def test_protocol_handshake_and_position_errors():
    async def run():
        protocol = UciProtocol(_engine())
        assert await protocol.handle_line("uci") == [
            "id name chessnets",
            "id author chessnets",
            "uciok",
        ]
        assert await protocol.handle_line("isready") == ["readyok"]
        errors = await protocol.handle_line("position fen invalid")
        assert errors[0].startswith("info string")

    asyncio.run(run())


def test_protocol_go_returns_a_legal_move():
    async def run():
        protocol = UciProtocol(_engine())
        await protocol.handle_line("position startpos")
        await protocol.handle_line("go wtime 300000 btime 300000")
        assert protocol._search_task is not None
        await protocol._search_task
        assert protocol.outputs[-1] == "bestmove e2e4"

    asyncio.run(run())


def test_uci_config_rejects_game_aware_metadata():
    with pytest.raises(ValueError, match="frozen"):
        UciEngineConfig("fake", metadata=MetadataConfig(mode="game"))


def test_go_values_are_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        UciGoParameters(wtime=-1)


def test_custom_fen_pgn_reaches_pipelines_at_the_right_position():
    from chessnets.utils.board_history import BoardHistory

    fen = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 4 20"
    position = UciPosition.from_fen(fen, ["e8d7", "e1e2"])

    history = BoardHistory.from_pgn(position.to_pgn())

    assert history.active_board.fen() == position.board.fen()
    assert history.positions[0].board.fen() == fen
