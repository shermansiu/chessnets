import torch

from chessnets.pipelines.carlini.pipeline_carlini import (
    CarliniOpenAIInstructChessPipeline,
)
from chessnets.pipelines.lc0.pipeline_lc0 import Lc0Pipeline
from chessnets.pipelines.maia2.pipeline_maia2 import Maia2Pipeline
from chessnets.pipelines.maia2 import pipeline_maia2
from chessnets.pipelines.text_generation.pipeline_text_generation import (
    HFTransformersProbabilityTextPipeline,
    OpenAIProbabilityTextPipeline,
)


class _Model(torch.nn.Module):
    pass


def test_lc0_from_pretrained_delegates_to_checkpoint_loader(monkeypatch):
    calls = {}
    model = _Model()

    def load(model_id, is_eval, device):
        calls.update(model_id=model_id, is_eval=is_eval, device=device)
        return model

    from chessnets.loaders import load_lc0_model

    monkeypatch.setattr(load_lc0_model, "load_pretrained_model", load)
    pipeline = Lc0Pipeline.from_pretrained("local-checkpoint", device="cpu")

    assert pipeline.model is model
    # The device reaches the loader so weights are read straight onto it.
    assert calls == {"model_id": "local-checkpoint", "is_eval": True, "device": "cpu"}


def test_maia2_from_pretrained_accepts_registry_alias_and_pipeline_options(monkeypatch):
    calls = {}
    model = _Model()

    def load(source, **kwargs):
        calls.update(source=source, **kwargs)
        return model

    class ModelLoader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            return load(source, **kwargs)

    monkeypatch.setattr(pipeline_maia2, "_load_maia2_model", lambda: ModelLoader)
    pipeline = Maia2Pipeline.from_pretrained(
        "maia2_rapid", device="cpu", temperature=0.25
    )

    assert pipeline.model is model
    assert pipeline.temperature == 0.25
    assert calls == {
        "source": "maia2_rapid",
        "time_control": "rapid",
        "device": "cpu",
        "revision": None,
        "cache_dir": None,
        "token": None,
        "local_files_only": False,
        "use_safetensors": None,
    }


def test_model_id_pipeline_loaders_construct_the_requested_subclass():
    class TextPipeline(HFTransformersProbabilityTextPipeline):
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.kwargs = kwargs

    text_pipeline = TextPipeline.from_pretrained("org/model", temperature=0.2)
    assert text_pipeline.model_id == "org/model"
    assert text_pipeline.kwargs == {"temperature": 0.2}

    class OpenAIPipeline(OpenAIProbabilityTextPipeline):
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.kwargs = kwargs

    openai_pipeline = OpenAIPipeline.from_pretrained("gpt-test")
    assert openai_pipeline.model_id == "gpt-test"

    class CarliniPipeline(CarliniOpenAIInstructChessPipeline):
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.kwargs = kwargs

    carlini_pipeline = CarliniPipeline.from_pretrained("text-test")
    assert carlini_pipeline.model_id == "text-test"
