from types import SimpleNamespace

import torch

from chessnets.pipelines.text_generation import pipeline_text_generation


def test_transformers_loader_forwards_checkpoint_options(monkeypatch):
    calls = {}
    tokenizer = object()

    class Model:
        def to(self, *, device):
            calls["device"] = device
            return self

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["tokenizer"] = (model_id, kwargs)
            return tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return Model()

    monkeypatch.setattr(
        pipeline_text_generation,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            AutoModelForCausalLM=AutoModelForCausalLM,
        ),
    )

    pipeline = object.__new__(pipeline_text_generation.TransformersProbabilityPipeline)
    model, loaded_tokenizer = pipeline.load_model_and_tokenizer(
        "org/checkpoint",
        device="cpu",
        dtype=torch.float16,
        revision="revision-1",
        cache_dir="cache",
        token="secret",
        local_files_only=True,
        tokenizer_kwargs={"use_fast": False},
        model_kwargs={"use_safetensors": True, "attn_implementation": "eager"},
    )

    assert isinstance(model, Model)
    assert loaded_tokenizer is tokenizer
    assert calls["device"] == "cpu"
    assert calls["tokenizer"] == (
        "org/checkpoint",
        {
            "revision": "revision-1",
            "cache_dir": "cache",
            "token": "secret",
            "local_files_only": True,
            "use_fast": False,
        },
    )
    assert calls["model"] == (
        "org/checkpoint",
        {
            "revision": "revision-1",
            "cache_dir": "cache",
            "token": "secret",
            "local_files_only": True,
            "use_safetensors": True,
            "attn_implementation": "eager",
            "dtype": torch.float16,
        },
    )
