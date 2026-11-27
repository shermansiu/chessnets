import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from chessnets.loaders.convert_maia2 import convert_maia2_checkpoint
from chessnets.models.maia2.configuration_maia2 import Maia2Config
from chessnets.models.maia2.modeling_maia2 import Maia2Model

TRAINING_CONFIG = """\
dim_cnn: 2
dim_vit: 16
num_blocks_cnn: 1
num_blocks_vit: 1
input_channels: 18
vit_length: 1
elo_dim: 4
side_info: true
value: true
batch_size: 8192
data_root: /not/a/runtime/path
"""


def _write_source(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(TRAINING_CONFIG, encoding="utf-8")
    config = Maia2Config.from_yaml(yaml_path)
    model = Maia2Model.from_config(config)
    state = {f"module.{key}": value for key, value in model.state_dict().items()}
    checkpoint = tmp_path / "blitz_model.pt"
    torch.save(
        {
            "model_state_dict": state,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "accumulated_games": 12,
            "accumulated_samples": 34,
        },
        checkpoint,
    )
    return checkpoint, yaml_path, model


def test_maia2_conversion_is_tensor_and_output_exact(tmp_path):
    checkpoint, yaml_path, original = _write_source(tmp_path)
    output = tmp_path / "converted"

    report = convert_maia2_checkpoint(
        checkpoint, yaml_path, output, time_control="blitz"
    )

    assert report["tensors_identical"] is True
    assert report["outputs_identical"] is True
    restored = load_file(output / "model.safetensors")
    assert restored.keys() == original.state_dict().keys()
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, restored[name])
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["time_control"] == "blitz"
    assert "batch_size" not in config["model"]
    assert "data_root" not in config["model"]


def test_maia2_loader_prefers_safetensors_and_validates_time_control(tmp_path):
    checkpoint, yaml_path, original = _write_source(tmp_path)
    output = tmp_path / "converted"
    convert_maia2_checkpoint(checkpoint, yaml_path, output, time_control="blitz")

    loaded = Maia2Model.from_pretrained(output)

    assert not loaded.training
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])
    with pytest.raises(ValueError, match="Requested 'rapid'"):
        Maia2Model.from_pretrained(output, time_control="rapid")


def test_maia2_legacy_model_pt_fallback_is_explicitly_safe(tmp_path):
    checkpoint, yaml_path, original = _write_source(tmp_path)
    config = Maia2Config.from_yaml(yaml_path)
    config.time_control = "blitz"
    config.save_pretrained(tmp_path)
    checkpoint.rename(tmp_path / "model.pt")

    loaded = Maia2Model.from_pretrained(tmp_path, use_safetensors=False)

    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])


def test_maia2_loader_rejects_non_exact_state(tmp_path):
    checkpoint, yaml_path, _ = _write_source(tmp_path)
    output = tmp_path / "converted"
    convert_maia2_checkpoint(checkpoint, yaml_path, output, time_control="blitz")
    state = load_file(output / "model.safetensors")
    state.pop("pos_embedding")
    save_file(state, output / "model.safetensors")

    with pytest.raises(RuntimeError, match="Missing key"):
        Maia2Model.from_pretrained(output)
