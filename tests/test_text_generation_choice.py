import re
import types

import chess
import pytest

from chessnets.pipelines.text_generation import pipeline_text_generation as text

NUMBERED_REGEX = re.compile(text.NUMBERED_MOVE_REGEX)
PGN_CONTINUATION_REGEX = re.compile(text.PGN_CONTINUATION_MOVE_REGEX)


def _history(board):
    return types.SimpleNamespace(active_board=board)


@pytest.mark.parametrize(
    "fen, uci, expected",
    [
        (chess.STARTING_FEN, "e2e4", ["1. e4"]),
        (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "g1f3",
            ["2. Nf3"],
        ),
        (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            "b8c6",
            ["2... Nc6"],
        ),
    ],
)
def test_numbered_choices_use_position_move_number_and_turn(fen, uci, expected):
    board = chess.Board(fen)
    choices = text.numbered_move_choices(_history(board), [chess.Move.from_uci(uci)])

    assert choices == expected
    assert all(NUMBERED_REGEX.fullmatch(choice) for choice in choices)


def test_pgn_header_choices_cover_all_regex_whitespace_and_number_variants():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    choices = text.pgn_continuation_move_choices(_history(board), [move])

    assert choices == ["e4", " e4", "1. e4", " 1. e4"]
    assert all(PGN_CONTINUATION_REGEX.fullmatch(choice) for choice in choices)


def test_pgn_header_choices_use_ellipsis_for_black():
    board = chess.Board()
    board.push_uci("e2e4")

    choices = text.pgn_continuation_move_choices(
        _history(board), [chess.Move.from_uci("e7e5")]
    )

    assert choices == ["e5", " e5", "1... e5", " 1... e5"]


def test_choice_forward_builds_one_generator_per_position(monkeypatch):
    generated = []

    class Choice:
        def __init__(self, items):
            self.items = items

    def generator(model, output_type):
        def generate(prompt, **kwargs):
            generated.append((model, output_type.items, prompt, kwargs))
            return output_type.items[0]

        return generate

    monkeypatch.setattr(
        text,
        "outlines",
        types.SimpleNamespace(
            Generator=generator,
            types=types.SimpleNamespace(Choice=Choice),
        ),
    )
    pipeline = object.__new__(text.TextGenerationPipeline)
    pipeline.pipeline_config = text.TextGenerationPipelineConfig(
        prompt_format="pgn_header", constraint="choice"
    )
    pipeline.model = "model"
    pipeline.temperature = 0.25
    white_board = chess.Board()
    black_board = chess.Board()
    black_board.push_uci("e2e4")

    results = pipeline.forward_batch(
        ["first", "second"],
        [_history(white_board), _history(black_board)],
        [
            [chess.Move.from_uci("e2e4")],
            [chess.Move.from_uci("e7e5")],
        ],
    )

    assert results == ["e4", "e5"]
    assert generated == [
        (
            "model",
            ["e4", " e4", "1. e4", " 1. e4"],
            "first",
            {"temperature": 0.25},
        ),
        (
            "model",
            ["e5", " e5", "1... e5", " 1... e5"],
            "second",
            {"temperature": 0.25},
        ),
    ]


def test_pipeline_rejects_unknown_constraint_before_model_loading():
    pipeline = object.__new__(text.TextGenerationPipeline)

    with pytest.raises(ValueError, match="constraint"):
        pipeline.__init__("unused", constraint="unknown")


def test_config_rejects_unknown_prompt_format():
    with pytest.raises(ValueError, match="prompt format"):
        text.TextGenerationPipelineConfig(prompt_format="unknown")


def test_fallback_move_is_the_same_in_every_run():
    import random

    board = chess.Board()
    history = _history(board)
    moves = list(board.legal_moves)
    picks = set()
    for seed in range(5):
        random.seed(seed)
        prediction = text.OutlinesProbabilityPipeline.post_process_failure(
            history, moves
        )
        picks.add(prediction.best_move_uci)
    assert picks == {"h2h4"}


def test_shuffled_legal_moves_are_the_same_in_every_run():
    import random

    from chessnets.utils import board_history

    history = board_history.BoardHistory.from_pgn("1. e4 e5")
    orders = set()
    for seed in range(5):
        random.seed(seed)
        orders.add(tuple(text._PromptDataset.history_to_legal_moves(history, True)))
    assert len(orders) == 1
    assert sorted(next(iter(orders)), key=str) == sorted(
        history.active_board.legal_moves, key=str
    )
