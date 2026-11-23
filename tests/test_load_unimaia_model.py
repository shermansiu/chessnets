import pathlib

import pytest
import torch

from chessnets.loaders.load_unimaia_model import (
    _load_local_training_checkpoint,
    _remap_old_t5gemma2_state_dict,
    load_unimaia_model,
)


def test_remap_old_t5gemma2_state_dict_updates_text_and_vision_prefixes():
    tensor = torch.tensor([1.0])
    state_dict = {
        "text_encoder.base_model.model.layers.0.weight": tensor,
        "text_encoder.base_model.model.embed_tokens.weight": tensor,
        "text_encoder.base_model.model.norm.weight": tensor,
        (
            "text_encoder.base_model.model.vision_tower.vision_model."
            "encoder.layers.0.weight"
        ): tensor,
        "fusion_layer.weight": tensor,
    }

    remapped = _remap_old_t5gemma2_state_dict(state_dict)

    assert set(remapped) == {
        "text_encoder.base_model.model.text_model.layers.0.weight",
        "text_encoder.base_model.model.text_model.embed_tokens.weight",
        "text_encoder.base_model.model.text_model.norm.weight",
        "text_encoder.base_model.model.vision_tower.encoder.layers.0.weight",
        "fusion_layer.weight",
    }
    assert all(value is tensor for value in remapped.values())


def test_load_local_training_checkpoint_extracts_only_model(tmp_path: pathlib.Path):
    path = tmp_path / "model_checkpoint_2013_12.pt"
    expected = {"layer.weight": torch.arange(4)}
    torch.save(
        {
            "model": expected,
            "optimizer": {"state": {0: {"momentum": torch.ones(4)}}},
            "num_training_steps": 12,
        },
        path,
    )

    actual = _load_local_training_checkpoint(path)

    assert actual.keys() == expected.keys()
    assert torch.equal(actual["layer.weight"], expected["layer.weight"])


def test_load_local_training_checkpoint_requires_model_mapping(
    tmp_path: pathlib.Path,
):
    path = tmp_path / "model_checkpoint_2013_12.pt"
    torch.save({"optimizer": {}}, path)

    with pytest.raises(TypeError, match="non-empty 'model' mapping"):
        _load_local_training_checkpoint(path)


def test_training_pt_requires_explicit_config(tmp_path: pathlib.Path):
    path = tmp_path / "model_checkpoint_2013_12.pt"
    torch.save({"model": {"layer.weight": torch.ones(1)}}, path)

    with pytest.raises(ValueError, match="resolved UniMaia config is required"):
        load_unimaia_model(path, checkpoint_format="training_pt")


def test_training_pt_rejects_remote_identifier():
    with pytest.raises(ValueError, match="existing local files"):
        load_unimaia_model(
            "organization/checkpoint.pt",
            config=object(),
            checkpoint_format="training_pt",
        )
