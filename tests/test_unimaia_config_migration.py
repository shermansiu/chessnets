import json

from chessnets.loaders.convert_unimaia_config import migrate_unimaia_config


def test_unimaia_config_migration_embeds_lc0_and_archives_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "lc0_model_id: local/lc0\n"
        "llm_model_id: local/text\n"
        "lc0_config_path: lc0_config.yaml\n"
        "model_path: model.safetensors\n",
        encoding="utf-8",
    )
    (tmp_path / "lc0_config.yaml").write_text(
        "model:\n  policy: attention\n  encoder_layers: 1\n",
        encoding="utf-8",
    )

    preview = migrate_unimaia_config(tmp_path)
    assert preview["canonical_config"]["lc0_config"]["model"]["policy"] == "attention"
    assert not (tmp_path / "config.json").exists()

    migrate_unimaia_config(tmp_path, write=True, archive_yaml=True)
    values = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert values["lc0_config_path"] is None
    assert values["lc0_config"]["model"]["encoder_layers"] == 1
    assert (tmp_path / "legacy" / "inference_config.yaml").is_file()
    assert (tmp_path / "training" / "original_lczero_training_config.yaml").is_file()
