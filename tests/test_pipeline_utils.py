import random

import pytest
import torch

from chessnets.pipelines import pipeline_utils


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def _collate(batch):
    return list(batch)


def test_resolve_num_workers_avoids_workers_for_small_inputs(monkeypatch):
    monkeypatch.setattr(pipeline_utils, "PREPROCESSING_WORKERS", 4)

    assert pipeline_utils.resolve_num_workers(0) == 0
    assert pipeline_utils.resolve_num_workers(32) == 0
    assert pipeline_utils.resolve_num_workers(33) == 4
    assert pipeline_utils.resolve_num_workers(1, requested_workers=2) == 2


@pytest.mark.parametrize("bad_value", [-1])
def test_resolve_num_workers_rejects_invalid_values(bad_value):
    with pytest.raises(ValueError):
        pipeline_utils.resolve_num_workers(bad_value)
    with pytest.raises(ValueError):
        pipeline_utils.resolve_num_workers(1, requested_workers=bad_value)
    with pytest.raises(ValueError):
        pipeline_utils.resolve_num_workers(1, parallel_threshold=bad_value)


def test_iter_dataset_batches_uses_local_collation_and_handles_partial_batch():
    batches = list(
        pipeline_utils.iter_dataset_batches(
            _Dataset(5), batch_size=2, collate_fn=_collate, num_workers=0
        )
    )

    assert batches == [[0, 1], [2, 3], [4]]


def test_iter_dataset_batches_preserves_shuffle_contract(monkeypatch):
    monkeypatch.setattr(random, "shuffle", lambda values: values.reverse())

    batches = list(
        pipeline_utils.iter_dataset_batches(
            _Dataset(4),
            batch_size=4,
            collate_fn=_collate,
            shuffle=True,
            num_workers=0,
        )
    )

    assert batches == [[3, 2, 1, 0]]


def test_iter_dataset_batches_rejects_persistent_workers_without_workers():
    with pytest.raises(ValueError, match="persistent_workers"):
        list(
            pipeline_utils.iter_dataset_batches(
                _Dataset(1),
                batch_size=1,
                collate_fn=_collate,
                persistent_workers=True,
            )
        )


def test_adaptive_process_map_is_inline_for_small_inputs(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("process pool should not be used")

    monkeypatch.setattr(pipeline_utils, "process_map", fail_if_called)

    result = pipeline_utils.adaptive_process_map(
        lambda value: value * 2,
        [1, 2, 3],
        parallel_threshold=3,
    )

    assert result == [2, 4, 6]


def test_adaptive_process_map_can_be_forced_inline(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("process pool should not be used")

    monkeypatch.setattr(pipeline_utils, "process_map", fail_if_called)

    result = pipeline_utils.adaptive_process_map(
        lambda value: value + 1,
        [1, 2, 3, 4],
        parallel_threshold=0,
        max_workers=0,
    )

    assert result == [2, 3, 4, 5]


def test_adaptive_process_map_uses_process_map_above_threshold(monkeypatch):
    calls = []

    def fake_process_map(function, *iterables, **kwargs):
        calls.append(kwargs)
        return list(map(function, *iterables))

    monkeypatch.setattr(pipeline_utils, "process_map", fake_process_map)

    result = pipeline_utils.adaptive_process_map(
        lambda value: value * 3,
        [1, 2, 3, 4],
        parallel_threshold=3,
        max_workers=2,
    )

    assert result == [3, 6, 9, 12]
    assert calls == [{"max_workers": 2}]


def test_adaptive_process_map_returns_empty_without_starting_pool(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("process pool should not be used")

    monkeypatch.setattr(pipeline_utils, "process_map", fail_if_called)

    assert pipeline_utils.adaptive_process_map(lambda value: value, []) == []


def test_position_random_depends_only_on_position_and_purpose():
    import subprocess
    import sys

    import chess

    board = chess.Board()
    moves = list(board.legal_moves)
    random.seed(1)
    first = pipeline_utils.position_random(board, "fallback-move").choice(moves)
    random.seed(2)
    second = pipeline_utils.position_random(board, "fallback-move").choice(moves)
    assert first == second
    # Pinned so a change to the derivation, which changes archived results,
    # is deliberate.
    assert first.uci() == "h2h4"

    code = (
        "import chess\n"
        "from chessnets.pipelines.pipeline_utils import position_random\n"
        "b = chess.Board()\n"
        "print(position_random(b, 'fallback-move').choice(list(b.legal_moves)))\n"
    )
    for hash_seed in ("0", "123"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": hash_seed},
        )
        assert out.stdout.strip() == "h2h4"

    shuffled = list(moves)
    pipeline_utils.position_random(board, "shuffle-legal-moves").shuffle(shuffled)
    assert sorted(shuffled, key=str) == sorted(moves, key=str)
