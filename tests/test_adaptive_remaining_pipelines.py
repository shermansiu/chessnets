from pathlib import Path

import pytest
import torch

from chessnets.models.maia2 import utils as maia2_utils
from chessnets.pipelines.carlini.pipeline_carlini import (
    CarliniOpenAIInstructChessPipeline,
    CarliniOpenAIInstructChessDataset,
)
from chessnets.pipelines.chess_clip.pipeline_chess_clip import ChessClipDataset
from chessnets.pipelines.maia2.pipeline_maia2 import Maia2Dataset, Maia2Pipeline
from chessnets.pipelines.text_generation.pipeline_text_generation import (
    DEFAULT_OUTLINES_PROMPT_TEMPLATE,
    OutlinesProbabilityTextPipeline,
)


PGN = "1. e4 e5 2. Nf3 *"


class _Maia2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def forward(self, board, elo_self, elo_oppo):
        return (
            torch.zeros((board.shape[0], len(maia2_utils.MOVES))),
            None,
            torch.zeros(board.shape[0]),
        )


def test_maia2_dataset_uses_local_small_input_path():
    dataset = Maia2Dataset(
        "8/8/8/8/8/8/8/K6k w - - 0 1",
        1500,
        1500,
    )

    assert len(dataset) == 1
    assert dataset[0][0].shape[0] == 18


def test_maia2_small_prediction_runs_without_workers():
    predictions = Maia2Pipeline(_Maia2Model()).make_prediction(
        "8/8/8/8/8/8/8/K6k w - - 0 1",
        1500,
        1500,
        num_workers=0,
    )

    assert len(predictions.predictions) == 1


def test_maia2_rejects_removed_singular_fen_keyword():
    with pytest.raises(ValueError, match="Unsupported pipeline input.*fen"):
        Maia2Pipeline(_Maia2Model()).make_prediction(
            fens="8/8/8/8/8/8/8/K6k w - - 0 1",
            elo_self=1500,
            elo_oppo=1500,
            fen="8/8/8/8/8/8/8/K6k w - - 0 1",
            num_workers=0,
        )


def test_chess_clip_dataset_uses_local_history_preparation():
    dataset = ChessClipDataset(
        PGN,
        "Describe the position",
        lambda texts: torch.zeros((len(texts), 1), dtype=torch.int64),
        num_workers=0,
    )

    assert len(dataset) == 1
    board_tensor, action_tensor, _, _, legal_moves, seq_length = dataset[0]
    assert board_tensor.shape[0] == action_tensor.shape[0] == len(legal_moves)
    assert seq_length == len(legal_moves)


class _FailingTextPipeline(OutlinesProbabilityTextPipeline):
    def __init__(self):
        self.prompt_template = DEFAULT_OUTLINES_PROMPT_TEMPLATE

    def forward(self, prompts):
        raise RuntimeError("synthetic backend failure")


def test_text_generation_failure_branch_keeps_one_row_per_input():
    predictions = _FailingTextPipeline().make_prediction(
        PGN,
        "Describe the position",
        batch_size=2,
        num_workers=0,
    )

    assert len(predictions.predictions) == 1
    assert predictions.predictions[0].raw_response is None


class _FakeCarlini(CarliniOpenAIInstructChessPipeline):
    def forward(self, prompt, num_tokens):
        return "e4"


def test_carlini_small_prediction_uses_local_preparation(tmp_path: Path):
    pipeline = _FakeCarlini.__new__(_FakeCarlini)
    pipeline.num_lookahead_tokens = 9
    pipeline.warn_on_game_over = True
    pipeline.pgn_header_override = None
    pipeline.model_id = "synthetic"
    pipeline.cache = {}
    pipeline.cache_path = tmp_path / "cache.json"

    predictions = pipeline.make_prediction(PGN, num_workers=0)

    assert len(predictions.predictions) == 1
    assert pipeline.cache_path.exists()


def test_carlini_dataset_accepts_empty_input_without_workers():
    dataset = CarliniOpenAIInstructChessDataset([], num_workers=0)

    assert len(dataset) == 0
