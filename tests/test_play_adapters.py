import asyncio
import types

from chessnets.pipelines.base import PipelineInputConfig
from chessnets.play.adapters import PipelinePlayAdapter
from chessnets.play.context import LiveGameState
from chessnets.play.metadata import ResolvedMetadata


class FakePipeline:
    input_config = PipelineInputConfig.from_inputs(required=("pgns",))

    def make_prediction(self, pgns, **kwargs):
        assert "[White" in pgns
        return types.SimpleNamespace(
            predictions=[types.SimpleNamespace(best_move_uci="e2e4")]
        )


class FakeFenPipeline:
    input_config = PipelineInputConfig.from_inputs(required=("fens",))

    def make_prediction(self, fens, **kwargs):
        raise NotImplementedError


def test_generic_adapter_calls_existing_batch_api_for_one_position():
    game = LiveGameState.from_game_full(
        {
            "id": "game",
            "variant": {"key": "standard"},
            "white": {"id": "bot", "name": "bot", "rating": 1500},
            "black": {"id": "opponent", "name": "opponent", "rating": 1300},
            "state": {"moves": "", "status": "started"},
        },
        "bot",
    )
    decision = asyncio.run(
        PipelinePlayAdapter().predict(FakePipeline(), game, ResolvedMetadata({}))
    )
    assert decision.move_uci == "e2e4"


def test_generic_adapter_uses_canonical_fens_keyword():
    game = LiveGameState.from_game_full(
        {
            "id": "game",
            "variant": {"key": "standard"},
            "white": {"id": "bot", "name": "bot", "rating": 1500},
            "black": {"id": "opponent", "name": "opponent", "rating": 1300},
            "state": {"moves": "", "status": "started"},
        },
        "bot",
    )

    kwargs = PipelinePlayAdapter()._build_kwargs(
        FakeFenPipeline(), game, ResolvedMetadata({})
    )

    assert kwargs["fens"] == game.fen
    assert "fen" not in kwargs
