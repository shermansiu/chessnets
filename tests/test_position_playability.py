import warnings

import chess
import pytest

from chessnets.pipelines.pipeline_utils import (
    DrawnPositionWarning,
    check_position_is_playable,
)


def _after(*moves, fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    return board


@pytest.mark.parametrize(
    ("board", "match"),
    [
        (chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"), "checkmate"),
        (chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), "stalemate"),
    ],
)
def test_finished_games_raise(board, match):
    with pytest.raises(ValueError, match=match):
        check_position_is_playable(board)


SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8"]


@pytest.mark.parametrize(
    ("board", "match"),
    [
        (_after(*SHUFFLE * 2), "threefold repetition"),
        (_after(*SHUFFLE * 4), "fivefold repetition"),
        (chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 100 80"), "50-move rule"),
        (chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 150 100"), "75-move rule"),
        (chess.Board("4k3/8/8/8/8/8/8/4KB2 w - - 0 1"), "insufficient material"),
    ],
)
def test_drawn_positions_warn(board, match):
    with pytest.warns(DrawnPositionWarning, match=match):
        check_position_is_playable(board)


def test_ordinary_positions_are_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_position_is_playable(_after(*SHUFFLE))  # Only twofold
        check_position_is_playable(chess.Board())
