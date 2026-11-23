import chess
import numpy as np
import pytest

from chessnets.utils.board_history import BoardHistory


def _history_with_moves(*moves: str, board: chess.Board | None = None) -> BoardHistory:
    history = BoardHistory(active_board=board if board is not None else chess.Board())
    for move in moves:
        history.append(chess.Move.from_uci(move))
    return history


def test_positions_are_stackless_snapshots():
    history = _history_with_moves(
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1b5",
    )

    assert len(history.active_board.move_stack) == len(history.moves)
    assert all(not position.board.move_stack for position in history.positions)

    # A historical board must remain unchanged when the active board advances.
    historical_fen = history.positions[2].board.fen()
    history.append(chess.Move.from_uci("g8f6"))
    assert history.positions[2].board.fen() == historical_fen


def test_copy_with_move_shares_snapshots_but_not_mutable_history():
    history = _history_with_moves("e2e4", "e7e5", "g1f3")
    fork = history.copy_with_move(chess.Move.from_uci("b8c6"))

    assert len(history) == 3
    assert len(fork) == 4
    assert fork.positions[: len(history.positions)] == history.positions
    assert fork.positions[-1] is not history.positions[-1]
    assert len(fork.active_board.move_stack) == len(fork.moves)
    assert all(not position.board.move_stack for position in fork.positions)


def test_asarray_only_uses_the_requested_eight_position_window():
    history = _history_with_moves(
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1b5",
        "a7a6",
        "b5a4",
        "g8f6",
        "e1g1",
        "f8e7",
    )

    final = history.asarray()
    second_last = history.asarray(-2)

    assert final.shape == (112, 64)
    assert final.dtype == np.float32
    np.testing.assert_array_equal(
        second_last, history.asarray(len(history.positions) - 2)
    )


def test_asarray_rejects_indices_outside_history():
    history = BoardHistory()

    with pytest.raises(IndexError):
        history.asarray(len(history.positions))
    with pytest.raises(IndexError):
        history.asarray(-len(history.positions) - 1)


def test_repetition_counts_initial_position_and_uses_absolute_plies():
    history = _history_with_moves(
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
    )

    assert history.positions[4].repetitions == 1
    assert history.positions[4].plies_since_prev_rep == 4
    assert history.positions[8].repetitions == 2
    assert history.positions[8].plies_since_prev_rep == 4


def test_repetition_distance_is_correct_for_nonstandard_fullmove_fen():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 10")
    history = _history_with_moves(
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        board=board,
    )

    assert history.positions[-1].repetitions == 1
    assert history.positions[-1].plies_since_prev_rep == 4


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Same pieces, different side to move.
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/8/4K3 b - - 0 1"),
        # Same pieces, different castling rights.
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w Kkq - 0 1",
        ),
        # A capturable en passant square distinguishes positions.
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1",
        ),
    ],
)
def test_position_key_distinguishes_what_fen_distinguishes(first, second):
    from chessnets.utils.board_history import position_key

    assert position_key(chess.Board(first)) != position_key(chess.Board(second))


def test_position_key_ignores_clocks_and_uncapturable_en_passant():
    from chessnets.utils.board_history import board_fen, position_key

    # After 1. e4 the FEN records e3, but no black pawn can capture there.
    after_e4 = chess.Board()
    after_e4.push_san("e4")
    same = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 7 42")
    assert board_fen(after_e4) == board_fen(same)
    assert position_key(after_e4) == position_key(same)


BLACK_TO_MOVE = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 4 20"


def test_from_pgn_starts_from_the_fen_header():
    pgn = f'[SetUp "1"]\n[FEN "{BLACK_TO_MOVE}"]\n\n20... Kd7 21. Ke2 Rh2+ *'
    history = BoardHistory.from_pgn(pgn)

    assert history.positions[0].board.fen() == BLACK_TO_MOVE
    assert [m.uci() for m in history.moves] == ["e8d7", "e1e2", "h8h2"]
    assert history.positions[0].is_black_turn
    assert history.move_indices == [history.move_index(i) for i in range(3)]


def test_to_pgn_round_trips_a_custom_starting_position():
    pgn = f'[SetUp "1"]\n[FEN "{BLACK_TO_MOVE}"]\n\n20... Kd7 21. Ke2 *'
    history = BoardHistory.from_pgn(pgn)

    restored = BoardHistory.from_pgn(history.to_pgn())

    assert restored.positions[0].board.fen() == BLACK_TO_MOVE
    assert restored.moves == history.moves
    assert np.array_equal(restored.asarray(), history.asarray())


def test_time_info_for_a_game_starting_with_black_to_move():
    pgn = (
        f'[SetUp "1"]\n[FEN "{BLACK_TO_MOVE}"]\n[TimeControl "60+0"]\n\n'
        "20... Kd7 { [%clk 0:00:58] } 21. Ke2 { [%clk 0:00:55] } "
        "21... Ke6 { [%clk 0:00:50] } *"
    )
    history = BoardHistory.from_pgn(pgn, should_load_clock_info=True)
    assert history.black_clock_times == [58, 50]
    assert history.white_clock_times == [55]

    # Same convention as games starting with White: at ply t, each side's
    # reading is the clock for its own move at or after t.
    info = history.get_time_info(0)  # Black to move
    assert info.active_player_time_remaining == 58
    assert info.opponent_time_remaining == 55
    assert info.active_player_current_move_time_spent == 8
    info = history.get_time_info(1)  # White to move
    assert info.active_player_time_remaining == 55
    assert info.opponent_time_remaining == 50
    assert info.opponent_last_move_time_spent == 8
    # Black's next move has no White reading after it yet.
    assert history.get_time_info(2) is None


def test_time_info_convention_matches_white_first_games():
    white_first = BoardHistory.from_pgn(
        '[TimeControl "60+0"]\n\n1. e4 { [%clk 0:00:58] } e5 { [%clk 0:00:55] } '
        "2. Nf3 { [%clk 0:00:50] } *",
        should_load_clock_info=True,
    )
    info = white_first.get_time_info(1)  # Black to move
    assert info.active_player_time_remaining == 55
    assert info.opponent_time_remaining == 50
