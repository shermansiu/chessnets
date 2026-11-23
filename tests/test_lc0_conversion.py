import json

from chessnets.loaders.convert_lc0 import convert_checkpoint
from chessnets.loaders import load_lc0_model
from chessnets.models.lc0_senet.configuration_lc0_senet import (
    LeelaInferenceConfig,
    LeelaNetModelConfig,
)
from chessnets.models.lc0_senet.modeling_lc0_senet import SeLeelaZeroNet


class _FakeProtobufAdapter:
    def __init__(self, config):
        self.config = config

    def inspect_checkpoint(self, path):
        return {
            "source_container_format": "lc0-protobuf",
            "source_weight_encoding": "LINEAR16",
            "decoded_tensor_dtype": "float32",
        }

    def load_model(self, model_path, config_path, *, is_eval=True):
        model = SeLeelaZeroNet.from_config(self.config)
        if is_eval:
            model.eval()
        return model


def test_converter_writes_portable_artifacts_and_sanitized_provenance(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "model.pb.gz").write_bytes(b"test protobuf placeholder")
    config = LeelaInferenceConfig(
        model=LeelaNetModelConfig(filters=8, residual_blocks=1, se_ratio=2)
    )
    (source / "config.yaml").write_text(
        "model:\n  filters: 8\n  residual_blocks: 1\n  se_ratio: 2\n",
        encoding="utf-8",
    )
    adapter = _FakeProtobufAdapter(config)
    monkeypatch.setattr(load_lc0_model, "_get_protobuf_adapter", lambda: adapter)

    report = convert_checkpoint(source, output)

    assert (output / "model.safetensors").is_file()
    assert (output / "config.json").is_file()
    assert (output / "original" / "model.pb.gz").is_file()
    assert (output / "training" / "original_lczero_training_config.yaml").is_file()
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_reference"] == str(source)
    assert "output" not in provenance
    assert report["tensor_count"] > 0
