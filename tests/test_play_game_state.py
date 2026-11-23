import chess

from chessnets.play.context import LiveGameState


def _game_full():
    return {
        "type": "gameFull",
        "id": "abc123",
        "variant": {"key": "standard"},
        "speed": "blitz",
        "rated": False,
        "initialFen": "startpos",
        "clock": {"initial": 300000, "increment": 3000},
        "white": {"id": "bot", "name": "bot", "rating": 1500},
        "black": {"id": "human", "name": "human", "rating": 1400},
        "state": {
            "type": "gameState",
            "moves": "e2e4 e7e5",
            "wtime": 299000,
            "btime": 300000,
            "winc": 3000,
            "binc": 3000,
            "status": "started",
        },
    }


def test_game_state_rebuilds_board_and_pgn():
    state = LiveGameState.from_game_full(_game_full(), "bot")
    assert state.ply == 2
    assert state.board.turn == chess.WHITE
    assert state.fen.startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3")
    assert 'TimeControl "300+3"' in state.to_pgn()
    assert state.bot_player.rating == 1500
    assert state.opponent.rating == 1400


def test_game_state_updates_are_idempotent():
    state = LiveGameState.from_game_full(_game_full(), "bot")
    state.apply_game_state(_game_full()["state"])
    state.apply_game_state(_game_full()["state"])
    assert state.moves == ["e2e4", "e7e5"]
    assert len(state.white_clock_history) == 1


def test_incremental_state_records_observed_move_time():
    state = LiveGameState.from_game_full(_game_full(), "bot")
    state.apply_game_state(
        {
            "moves": "e2e4 e7e5 g1f3",
            "wtime": 294000,
            "btime": 298000,
            "winc": 3000,
            "binc": 3000,
            "status": "started",
        }
    )
    assert state.move_seconds == [0, 0, 8]
    assert state.white_clock_history == [299, 294]
    assert state.black_clock_history == [300]


def test_reconnect_snapshot_pads_unobserved_clock_history():
    event = _game_full()
    event["state"] = {
        **event["state"],
        "moves": "e2e4 e7e5 g1f3 b8c6",
        "wtime": 294000,
        "btime": 296000,
    }

    state = LiveGameState.from_game_full(event, "bot")

    assert state.white_clock_history == [None, 294]
    assert state.black_clock_history == [None, 296]
