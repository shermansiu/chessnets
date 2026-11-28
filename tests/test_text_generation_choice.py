import re
import types
import warnings

import chess
import pytest

from chessnets.pipelines.text_generation import pipeline_text_generation as text
from chessnets.utils import board_history

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


def _install_choice_fakes(monkeypatch, generated, backends):
    class Choice:
        def __init__(self, items):
            self.items = items

    class Backend:
        def __init__(self, model):
            backends.append(model)

        def get_regex_logits_processor(self, regex):
            return regex

    def generator(model, *, processor):
        def generate(prompt, **kwargs):
            generated.append((model, processor, prompt, kwargs))
            return processor[0]

        return generate

    monkeypatch.setattr(
        text,
        "outlines",
        types.SimpleNamespace(
            Generator=generator,
            backends=types.SimpleNamespace(OutlinesCoreBackend=Backend),
            types=types.SimpleNamespace(
                Choice=Choice,
                dsl=types.SimpleNamespace(to_regex=lambda term: term.items),
            ),
        ),
    )


def test_choice_forward_builds_one_generator_per_position(monkeypatch):
    generated = []
    backends = []
    _install_choice_fakes(monkeypatch, generated, backends)
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
    # The vocabulary is converted once, not once per position.
    assert backends == ["model"]
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


def test_choice_warns_once_that_batch_size_is_not_used():
    pipeline = object.__new__(text.TextGenerationPipeline)

    with pytest.warns(UserWarning, match="one position at a time"):
        pipeline._warn_choice_batch_size(16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline._warn_choice_batch_size(16)
        object.__new__(text.TextGenerationPipeline)._warn_choice_batch_size(1)


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


def test_batched_choice_needs_the_choice_constraint():
    with pytest.raises(ValueError, match="batched_choice"):
        text.TextGenerationPipelineConfig(batched_choice=True)
    config = text.TextGenerationPipelineConfig(constraint="choice", batched_choice=True)
    assert config.batched_choice


def test_batched_choice_generates_the_batch_in_one_call():
    pipeline = object.__new__(text.TextGenerationPipeline)
    pipeline.pipeline_config = text.TextGenerationPipelineConfig(
        prompt_format="pgn_header",
        constraint="choice",
        batched_choice=True,
    )
    pipeline.temperature = 0.25
    calls = []

    class Generator:
        def batch(self, prompts, **kwargs):
            calls.append((prompts, kwargs))
            return [row[0] for row in self.choices]

    def batched_choice_generator(choices):
        generator = Generator()
        generator.choices = choices
        return generator

    pipeline._batched_choice_generator = batched_choice_generator
    black_board = chess.Board()
    black_board.push_uci("e2e4")

    results = pipeline.forward_batch(
        ["first", "second"],
        [_history(chess.Board()), _history(black_board)],
        [[chess.Move.from_uci("e2e4")], [chess.Move.from_uci("e7e5")]],
    )

    assert results == ["e4", "e5"]
    assert calls == [(["first", "second"], {"temperature": 0.25})]


def test_sort_by_length_batches_longest_first_and_keeps_input_order():
    pipeline = object.__new__(text.TextGenerationPipeline)
    pipeline.pipeline_config = text.TextGenerationPipelineConfig(
        prompt_format="pgn_header", sort_by_length=True
    )
    pipeline.input_config = pipeline.pipeline_config.input_config
    pipeline.prompt_template = pipeline.pipeline_config.default_prompt_template
    pipeline.model = types.SimpleNamespace()  # no tokenizer: sort by characters
    batches = []

    def forward_batch(prompts, histories, legal_moves):
        batches.append([len(prompt) for prompt in prompts])
        # Answer each position with its first legal move, so the answers
        # identify the positions.
        return [
            history.active_board.san(moves[0])
            for history, moves in zip(histories, legal_moves)
        ]

    pipeline.forward_batch = forward_batch
    pgns = ["1. e4", "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6", "1. d4 d5 2. c4", ""]

    predictions = pipeline.make_prediction(
        pgns, prompt=["p"] * len(pgns), batch_size=2, num_workers=0
    ).predictions

    lengths = [length for batch in batches for length in batch]
    assert lengths == sorted(lengths, reverse=True)
    expected = []
    for pgn in pgns:
        board = board_history.BoardHistory.from_pgn(pgn).active_board
        expected.append(board.san(next(iter(board.legal_moves))))
    assert [p.best_move_san for p in predictions] == expected
