import pytest
import torch

from chessnets.pipelines.base import Pipeline
from chessnets.pipelines.pipeline_utils import is_out_of_memory_error


class FittingPipeline(Pipeline):
    """Runs out of memory above ``limit`` items per batch."""

    auto_batch_sizes = {"cpu": 64, "cuda": 1024}

    def __init__(self, limit, device="cpu"):
        super().__init__()
        self.limit = limit
        self.device = device
        self.sizes = []

    def forward(self):
        pass

    def make_prediction(self, pgns, batch_size: int = 16):
        self.sizes.append(batch_size)
        if batch_size > self.limit:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        return len(pgns)


def test_auto_halves_until_a_batch_fits_and_keeps_that_size():
    pipeline = FittingPipeline(limit=200, device="cuda:0")

    assert pipeline(["1. e4"], batch_size="auto") == 1
    assert pipeline.sizes == [1024, 512, 256, 128]
    assert pipeline.last_batch_size == 128

    pipeline(["1. e4"], batch_size="auto")
    assert pipeline.sizes[-1] == 128


def test_auto_starts_from_the_devices_size():
    pipeline = FittingPipeline(limit=10_000, device="mps")
    pipeline(["1. e4"], batch_size="auto")
    assert pipeline.sizes == [64]


def test_explicit_sizes_are_used_as_given():
    pipeline = FittingPipeline(limit=10)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        pipeline(["1. e4"], batch_size=32)
    assert pipeline.sizes == [32]
    pipeline(["1. e4"], batch_size=4)
    assert pipeline.last_batch_size == 4


def test_auto_gives_up_at_one_item():
    pipeline = FittingPipeline(limit=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        pipeline(["1. e4"], batch_size="auto")
    assert pipeline.sizes[-1] == 1


def test_other_errors_are_not_retried():
    class Broken(FittingPipeline):
        def make_prediction(self, pgns, batch_size: int = 16):
            self.sizes.append(batch_size)
            raise ValueError("bad input")

    pipeline = Broken(limit=0)
    with pytest.raises(ValueError):
        pipeline(["1. e4"], batch_size="auto")
    assert pipeline.sizes == [64]


def test_pipelines_without_a_batch_size_ignore_auto():
    class Unbatched(FittingPipeline):
        def make_prediction(self, pgns):
            return "done"

    assert Unbatched(limit=0)(["1. e4"], batch_size="auto") == "done"


def test_out_of_memory_errors_are_recognised_through_their_causes():
    try:
        try:
            raise RuntimeError("MPS backend out of memory")
        except RuntimeError as error:
            raise ValueError("wrapped") from error
    except ValueError as wrapped:
        assert is_out_of_memory_error(wrapped)
    assert not is_out_of_memory_error(RuntimeError("shape mismatch"))
