import pytest
import torch

outlines_core = pytest.importorskip("outlines_core")
pytest.importorskip("outlines")

from chessnets.pipelines.text_generation.batched_choice import (  # noqa: E402
    PerRowChoiceLogitsProcessor,
)

EOS = 4
VOCAB = {"a": [0], "b": [1], "c": [2], "d": [3]}


def _index(regex):
    return outlines_core.Index(regex, outlines_core.Vocabulary(EOS, VOCAB))


def _allowed(logits):
    return [torch.isfinite(row).nonzero().flatten().tolist() for row in logits]


def test_each_row_follows_its_own_choices():
    processor = PerRowChoiceLogitsProcessor([_index("(ab)"), _index("(cd)")])
    input_ids = torch.tensor([[9], [9]])

    first = processor(input_ids, torch.zeros(2, 5))
    assert _allowed(first) == [[0], [2]]

    input_ids = torch.cat([input_ids, torch.tensor([[0], [2]])], dim=1)
    second = processor(input_ids, torch.zeros(2, 5))
    assert _allowed(second) == [[1], [3]]

    input_ids = torch.cat([input_ids, torch.tensor([[1], [3]])], dim=1)
    assert _allowed(processor(input_ids, torch.zeros(2, 5))) == [[EOS], [EOS]]


def test_finished_rows_stay_finished_while_others_generate():
    processor = PerRowChoiceLogitsProcessor([_index("(a)"), _index("(cd)")])
    input_ids = torch.tensor([[9], [9]])
    processor(input_ids, torch.zeros(2, 5))
    input_ids = torch.cat([input_ids, torch.tensor([[0], [2]])], dim=1)
    processor(input_ids, torch.zeros(2, 5))
    # Row 0 emitted EOS and is padded with it; row 1 continues.
    input_ids = torch.cat([input_ids, torch.tensor([[EOS], [3]])], dim=1)
    assert _allowed(processor(input_ids, torch.zeros(2, 5))) == [[EOS], [EOS]]


def test_reset_starts_the_guides_again_and_batch_size_is_checked():
    processor = PerRowChoiceLogitsProcessor([_index("(ab)")])
    processor(torch.tensor([[9]]), torch.zeros(1, 5))
    processor.reset()
    assert _allowed(processor(torch.tensor([[9]]), torch.zeros(1, 5))) == [[0]]

    with pytest.raises(ValueError, match="batch of 1"):
        PerRowChoiceLogitsProcessor([_index("(ab)")])(
            torch.tensor([[9], [9]]), torch.zeros(2, 5)
        )
