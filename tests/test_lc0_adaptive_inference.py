from types import SimpleNamespace

import torch

from chessnets.pipelines.lc0.pipeline_lc0 import Lc0Dataset, Lc0Pipeline
from chessnets.utils import decode_policy


PGN = "1. e4 e5 2. Nf3 *"


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.calls = 0

    def forward(self, board_tensor):
        self.calls += 1
        batch_size = board_tensor.shape[0]
        return SimpleNamespace(
            policy=torch.zeros((batch_size, len(decode_policy.MOVES))),
            value=torch.zeros((batch_size, 3)),
        )


def test_lc0_dataset_combines_history_and_array_preparation(monkeypatch):
    import chessnets.pipelines.lc0.pipeline_lc0 as module

    calls = []
    original = module.adaptive_process_map

    def spy(function, *iterables, **kwargs):
        calls.append((function, kwargs["max_workers"]))
        return original(function, *iterables, **kwargs)

    monkeypatch.setattr(module, "adaptive_process_map", spy)
    dataset = Lc0Dataset([PGN], num_workers=0)

    assert len(dataset) == 1
    assert dataset[0][0].shape == (112, 64)
    assert len(calls) == 1
    assert calls[0][1] == 0


def test_lc0_small_request_uses_single_model_batch():
    model = _Model()
    predictions = Lc0Pipeline(model).make_prediction(
        PGN,
        batch_size=16,
        num_workers=0,
    )

    assert len(predictions.predictions) == 1
    assert model.calls == 1


def test_lc0_empty_request_returns_empty_prediction_dataset():
    model = _Model()
    predictions = Lc0Pipeline(model).make_prediction([], num_workers=0)

    assert predictions.predictions == []
    assert model.calls == 0


def test_lc0_shuffle_preserves_input_order(monkeypatch):
    import chessnets.pipelines.pipeline_utils as pipeline_utils

    monkeypatch.setattr(
        pipeline_utils.random, "shuffle", lambda values: values.reverse()
    )
    model = _Model()
    predictions = Lc0Pipeline(model).make_prediction(
        ["1. e4 *", "1. d4 *"],
        shuffle=True,
        return_histories=True,
        num_workers=0,
    )

    assert [
        prediction.board_history.moves[0].uci()
        for prediction in predictions.predictions
    ] == ["e2e4", "d2d4"]


def test_lc0_post_processes_each_batch_without_retaining_all_outputs(monkeypatch):
    model = _Model()
    model_calls_during_post_process = []
    original = Lc0Pipeline.post_process_predictions

    def spy(*args, **kwargs):
        model_calls_during_post_process.append(model.calls)
        return original(*args, **kwargs)

    monkeypatch.setattr(Lc0Pipeline, "post_process_predictions", staticmethod(spy))
    Lc0Pipeline(model).make_prediction([PGN, PGN, PGN], batch_size=2, num_workers=0)

    assert model_calls_during_post_process == [1, 1, 2]
