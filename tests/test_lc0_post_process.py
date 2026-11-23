import chess
import pytest
import torch

from chessnets.pipelines.lc0.pipeline_lc0 import Lc0Pipeline
from chessnets.utils import decode_policy
from chessnets.utils.board_history import BoardHistory


def _predict(fen, policy=None):
    history = BoardHistory(active_board=chess.Board(fen))
    if policy is None:
        policy = torch.zeros(len(decode_policy.MOVES))
    result = torch.tensor([0.5, 0.3, 0.2])
    return Lc0Pipeline.post_process_predictions(
        policy, result, result[0] - result[2], None, history, 1.0
    )


@pytest.mark.parametrize(
    "fen", ["8/4P3/8/8/8/8/k7/4K3 w - - 0 1", "4k3/8/8/8/8/8/4p3/K7 b - - 0 1"]
)
def test_predictions_are_the_legal_moves_including_knight_promotions(fen):
    board = chess.Board(fen)
    prediction = _predict(fen)

    assert set(prediction.preds_list) == set(board.legal_moves)
    promotions = {m.uci()[-1] for m in prediction.preds_list if m.promotion}
    assert promotions == {"q", "r", "b", "n"}
    assert sum(prediction.preds_list.values()) == pytest.approx(1.0)


def test_knight_promotion_can_be_the_best_move():
    fen = "8/4P3/8/8/8/8/k7/4K3 w - - 0 1"
    policy = torch.full((len(decode_policy.MOVES),), -10.0)
    # Lc0 encodes the knight underpromotion as the plain pawn push.
    policy[decode_policy.MOVE_TO_INDEX["e7e8"]] = 10.0

    prediction = _predict(fen, policy)

    assert prediction.best_move_uci == "e7e8n"
    assert prediction.best_move_san == "e8=N"


def test_black_moves_are_reported_from_the_board_perspective():
    fen = "4k3/8/8/8/8/8/4p3/K7 b - - 0 1"
    policy = torch.full((len(decode_policy.MOVES),), -10.0)
    # Black's e2e1q is e7e8q from the side to move's (mirrored) perspective.
    policy[decode_policy.MOVE_TO_INDEX["e7e8q"]] = 10.0

    assert _predict(fen, policy).best_move_uci == "e2e1q"


def test_game_over_raises_a_clear_error():
    with pytest.raises(ValueError, match="game is over"):
        _predict("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")


def test_stalemate_raises_a_clear_error():
    with pytest.raises(ValueError, match="stalemate"):
        _predict("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")


def test_repetition_warns_but_still_predicts():
    from chessnets.pipelines import DrawnPositionWarning

    history = BoardHistory()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        history.append(chess.Move.from_uci(move))
    policy = torch.zeros(len(decode_policy.MOVES))
    result = torch.tensor([0.5, 0.3, 0.2])

    with pytest.warns(DrawnPositionWarning, match="threefold repetition"):
        prediction = Lc0Pipeline.post_process_predictions(
            policy, result, result[0] - result[2], None, history, 1.0
        )
    assert prediction.best_move_uci
