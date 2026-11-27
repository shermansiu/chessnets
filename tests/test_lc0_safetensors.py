import json

import pytest
import torch
from safetensors.torch import save_file

from chessnets.loaders.load_lc0_model import load_pretrained_model
from chessnets.loaders.safetensors import load_safetensors_file
from chessnets.models.lc0.configuration_lc0 import (
    Lc0InferenceConfig,
    Lc0ModelConfig,
    Lc0Policy,
)
from chessnets.models.lc0.modeling_lc0_seresnet import Lc0SeResNetModel


def _small_config():
    return Lc0InferenceConfig(
        model=Lc0ModelConfig(
            filters=8,
            residual_blocks=1,
            se_ratio=2,
            policy=Lc0Policy.convolution,
        )
    )


def test_lc0_safetensors_round_trip_is_strict_and_json_first(tmp_path):
    config = _small_config()
    model = Lc0SeResNetModel.from_config(config)
    config.save_pretrained(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "model:\n  filters: 16\n  residual_blocks: 2\n", encoding="utf-8"
    )
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    restored = load_pretrained_model(tmp_path)

    assert isinstance(restored, Lc0SeResNetModel)
    assert restored.config.model.filters == 8
    assert not restored.training
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)


def test_policy_mapping_is_not_checkpoint_state():
    model = Lc0SeResNetModel.from_config(_small_config())
    assert "policy_head.policy_indices" not in model.state_dict()
    assert model.policy_head.policy_indices.shape == (1858,)


def test_sharded_safetensors_index_is_validated(tmp_path):
    save_file({"a": torch.ones(2)}, tmp_path / "part-1.safetensors")
    save_file({"b": torch.zeros(3)}, tmp_path / "part-2.safetensors")
    index = {
        "weight_map": {
            "a": "part-1.safetensors",
            "b": "part-2.safetensors",
        }
    }
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps(index), encoding="utf-8")

    state = load_safetensors_file(path)
    assert set(state) == {"a", "b"}

    index["weight_map"]["a"] = "../escape.safetensors"
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe"):
        load_safetensors_file(path)


def test_strict_load_rejects_missing_tensor(tmp_path):
    config = _small_config()
    config.save_pretrained(tmp_path)
    model = Lc0SeResNetModel.from_config(config)
    state = model.state_dict()
    state.pop(next(iter(state)))
    save_file(state, tmp_path / "model.safetensors")

    with pytest.raises(RuntimeError, match="Missing key"):
        load_pretrained_model(tmp_path)


def test_build_with_state_dict_skips_init_and_keeps_policy_map():
    from chessnets.loaders.safetensors import build_with_state_dict

    reference = Lc0SeResNetModel.from_config(_small_config())
    model = build_with_state_dict(
        lambda: Lc0SeResNetModel.from_config(_small_config()), reference.state_dict()
    )

    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], tensor)
    assert not any(t.is_meta for t in (*model.parameters(), *model.buffers()))
    torch.testing.assert_close(
        model.policy_head.policy_indices, reference.policy_head.policy_indices
    )


def test_build_with_state_dict_rejects_factory_buffers_missing_from_checkpoint():
    from chessnets.loaders.safetensors import build_with_state_dict

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("table", torch.arange(3), persistent=False)

    state = {k: v for k, v in Model().state_dict().items()}
    with pytest.raises(RuntimeError, match="table"):
        build_with_state_dict(Model, state)
