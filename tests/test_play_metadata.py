from chessnets.play.config import MetadataConfig
from chessnets.play.context import LiveGameState
from chessnets.play.metadata import (
    FrozenMetadataResolver,
    GameAwareMetadataResolver,
)


def _state():
    return LiveGameState.from_game_full(
        {
            "id": "game",
            "variant": {"key": "standard"},
            "clock": {"initial": 300000, "increment": 3000},
            "white": {"id": "bot", "name": "bot", "rating": 1500},
            "black": {"id": "opponent", "name": "opponent", "rating": 1300},
            "state": {
                "moves": "",
                "wtime": 300000,
                "btime": 300000,
                "status": "started",
            },
        },
        "bot",
    )


def test_frozen_metadata_maps_bot_and_opponent_to_colors():
    result = FrozenMetadataResolver(
        MetadataConfig(
            values={
                "bot_elo": 1600,
                "opponent_elo": 1200,
                "time_control": "300+3",
            }
        )
    ).resolve(_state())
    assert result.values["white_elos"] == 1600
    assert result.values["black_elos"] == 1200
    assert result.values["time_controls"] == "300+3"


def test_game_aware_metadata_uses_live_ratings_and_actual_control():
    result = GameAwareMetadataResolver(
        MetadataConfig(mode="game", fallbacks={"bot_elo": 1000})
    ).resolve(_state())
    assert result.values["elo_self"] == 1500
    assert result.values["elo_opponent"] == 1300
    assert result.values["white_elos"] == 1500
    assert result.values["time_controls"] == "300+3"
