import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from chessnets.loaders.convert_unimaia_mergekit import (
    INDEX_FILENAME,
    POLICY_MAP_KEYS,
    inspect_mergekit_checkpoint,
    normalize_mergekit_checkpoint,
)
from chessnets.loaders.safetensors import load_safetensors_file
from chessnets.models.unimaia.configuration_unimaia import (
    TextEncoderType,
    UniMaiaConfig,
    VLMModelArchitectureType,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(tmp_path, *, policy_dtype=torch.bfloat16):
    source = tmp_path / "mergekit"
    source.mkdir()
    policy = {
        key: torch.arange(1858, dtype=torch.float32).to(policy_dtype)
        for key in sorted(POLICY_MAP_KEYS)
    }
    first = {"board.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)}
    first.update(dict(list(policy.items())[:3]))
    second = {"text.weight": torch.ones((2, 5), dtype=torch.bfloat16)}
    second.update(dict(list(policy.items())[3:]))
    save_file(first, source / "model-00001-of-00002.safetensors")
    save_file(second, source / "model-00002-of-00002.safetensors")
    weight_map = {
        **{key: "model-00001-of-00002.safetensors" for key in first},
        **{key: "model-00002-of-00002.safetensors" for key in second},
    }
    total_size = sum(value.numel() * value.element_size() for value in first.values())
    total_size += sum(value.numel() * value.element_size() for value in second.values())
    (source / INDEX_FILENAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size, "mergekit_version": "0.1.4"},
                "weight_map": weight_map,
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "phase1.yaml"
    config.write_text(
        "lc0_bt_model_id: local/lc0\n"
        "text_encoder_type: lora\n"
        "vlm_architecture_type: controlnet_controllable_policy_head\n"
        "lora_rank: 16\n"
        "use_dora: false\n",
        encoding="utf-8",
    )
    lc0_config = tmp_path / "lc0_config.json"
    lc0_config.write_text(
        json.dumps(
            {
                "architectures": ["Lc0ChessFormerModel"],
                "config_schema_version": 1,
                "model_type": "lc0",
                "model": {"policy": "attention"},
            }
        ),
        encoding="utf-8",
    )
    text_model = tmp_path / "text_model"
    text_model.mkdir()
    (text_model / "config.json").write_text(
        json.dumps({"model_type": "gpt_neox", "hidden_size": 16}), encoding="utf-8"
    )
    (text_model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (text_model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return source, config, lc0_config, text_model


def test_mergekit_normalization_reuses_shards_and_filters_policy_maps(tmp_path):
    source, training_config, lc0_config, text_model = _write_source(tmp_path)
    original_hashes = {path.name: _sha256(path) for path in source.iterdir()}
    output = tmp_path / "normalized"

    report = normalize_mergekit_checkpoint(
        source,
        training_config,
        output,
        lc0_config=lc0_config,
        text_model_directory=text_model,
        expected_tensor_count=8,
    )

    assert report["source"]["tensor_count"] == 8
    assert report["output"]["tensor_count"] == 2
    assert report["source"]["total_size"] - report["output"]["total_size"] == (
        6 * 1858 * 2
    )
    for shard in report["source"]["shards"]:
        assert (output / shard["name"]).is_symlink()
        assert _sha256(output / shard["name"]) == shard["sha256"]
    assert {path.name: _sha256(path) for path in source.iterdir()} == original_hashes

    normalized = json.loads((output / INDEX_FILENAME).read_text())
    assert set(normalized["weight_map"]) == {"board.weight", "text.weight"}
    assert normalized["metadata"]["chessnets_normalization"] == (
        "drop-nonpersistent-policy-maps-v1"
    )
    state = load_safetensors_file(output / INDEX_FILENAME)
    assert set(state) == {"board.weight", "text.weight"}

    config = UniMaiaConfig.from_json(output / "config.json")
    assert config.lc0_model_id == "local/lc0"
    assert config.lc0_config is not None
    assert config.lc0_config["architectures"] == ["Lc0ChessFormerModel"]
    assert config.lc0_config_path is None
    assert config.tokenizer_path == "."
    assert config.llm_config_path == "text_model"
    assert config.llm_model_id == "Waterhorse/chessgpt-base-v1"
    assert config.text_encoder_type is TextEncoderType.lora
    assert config.vlm_architecture_type is (
        VLMModelArchitectureType.controlnet_controllable_policy_head
    )
    assert config.model_path == INDEX_FILENAME
    assert not config.use_dora
    assert (output / "legacy" / "training_config.yaml").is_file()
    assert (output / "legacy" / "lc0_config.json").is_file()
    assert (output / "tokenizer.json").is_file()
    assert (output / "text_model" / "config.json").is_file()
    assert report["source"]["lc0_config_sha256"] == _sha256(lc0_config)

    reused = normalize_mergekit_checkpoint(
        source,
        training_config,
        output,
        lc0_config=lc0_config,
        text_model_directory=text_model,
        expected_tensor_count=8,
    )
    assert reused["reused"] is True


def test_mergekit_inspection_rejects_wrong_policy_dtype(tmp_path):
    source, _, _, _ = _write_source(tmp_path, policy_dtype=torch.int64)

    with pytest.raises(ValueError, match="Unexpected legacy policy map"):
        inspect_mergekit_checkpoint(source, expected_tensor_count=8)


def test_mergekit_inspection_rejects_unindexed_tensor(tmp_path):
    source, _, _, _ = _write_source(tmp_path)
    index_path = source / INDEX_FILENAME
    index = json.loads(index_path.read_text())
    index["weight_map"].pop("board.weight")
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 8 indexed tensors"):
        inspect_mergekit_checkpoint(source, expected_tensor_count=8)


def test_mergekit_normalization_refuses_unrelated_output(tmp_path):
    source, training_config, lc0_config, text_model = _write_source(tmp_path)
    output = tmp_path / "normalized"
    output.mkdir()
    (output / "provenance.json").write_text(
        json.dumps({"source": {"index_sha256": "different"}}), encoding="utf-8"
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        normalize_mergekit_checkpoint(
            source,
            training_config,
            output,
            lc0_config=lc0_config,
            text_model_directory=text_model,
            expected_tensor_count=8,
        )


def test_mergekit_normalization_rejects_reuse_with_different_lc0_config(tmp_path):
    source, training_config, lc0_config, text_model = _write_source(tmp_path)
    output = tmp_path / "normalized"
    normalize_mergekit_checkpoint(
        source,
        training_config,
        output,
        lc0_config=lc0_config,
        text_model_directory=text_model,
        expected_tensor_count=8,
    )
    lc0_config.write_text(
        lc0_config.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        normalize_mergekit_checkpoint(
            source,
            training_config,
            output,
            lc0_config=lc0_config,
            text_model_directory=text_model,
            expected_tensor_count=8,
        )
