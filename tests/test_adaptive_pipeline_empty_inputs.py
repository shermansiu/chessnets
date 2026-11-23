import torch

from chessnets.pipelines.allie.pipeline_allie import AlliePipeline
from chessnets.pipelines.unimaia.pipeline_unimaia import UniMaiaPipeline


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32


class _Tokenizer:
    def __len__(self):
        return 1


def test_allie_empty_request_returns_empty_dataset():
    predictions = AlliePipeline(_Model(), _Tokenizer()).make_prediction(
        [], [], [], [], num_workers=0
    )

    assert predictions.predictions == []


def test_unimaia_empty_request_returns_empty_dataset():
    predictions = UniMaiaPipeline(_Model(), _Tokenizer()).make_prediction(
        [], [], num_workers=0
    )

    assert predictions.predictions == []
