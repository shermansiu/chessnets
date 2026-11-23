import pytest

from chessnets.models.unimaia.configuration_unimaia import (
    TextEncoderType,
    UniMaiaConfig,
)


def test_legacy_training_fields_are_ignored():
    config = UniMaiaConfig.from_dict(
        {
            "lc0_bt_model_id": "local/lc0",
            "llm_model_id": "local/text",
            "text_encoder_type": "lora",
            "batch_size": 32,
            "optimizer": "adamw",
            "legacy_run_id": "ignored",
        }
    )

    assert config.lc0_model_id == "local/lc0"
    assert config.text_encoder_type is TextEncoderType.lora
    assert not hasattr(config, "batch_size")


def test_missing_model_fields_are_reported():
    with pytest.raises(ValueError, match="lc0_model_id.*llm_model_id"):
        UniMaiaConfig.from_dict({})


def test_legacy_outcome_count_can_use_default_labels():
    config = UniMaiaConfig.from_dict(
        {
            "lc0_model_id": "local/lc0",
            "llm_model_id": "local/text",
            "num_outcome_types": 11,
        }
    )
    assert len(config.outcome_labels) == 11


def test_bundled_text_config_path_is_retained():
    config = UniMaiaConfig.from_dict(
        {
            "lc0_model_id": "local/lc0",
            "llm_model_id": "local/text",
            "llm_config_path": "text_config.json",
            "tokenizer_path": "tokenizer",
        }
    )
    assert config.llm_config_path == "text_config.json"
    assert config.tokenizer_path == "tokenizer"


def test_canonical_json_round_trip_can_embed_lc0_config(tmp_path):
    config = UniMaiaConfig.from_dict(
        {
            "lc0_model_id": "local/lc0",
            "llm_model_id": "local/text",
            "lc0_config": {"model": {"policy": "attention"}},
        }
    )
    path = config.save_pretrained(tmp_path)
    restored = UniMaiaConfig.from_json(path)

    assert restored == config
    assert '"model_type": "unimaia"' in path.read_text(encoding="utf-8")
