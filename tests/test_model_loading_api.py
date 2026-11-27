from types import SimpleNamespace

from chessnets.models.allie.configuration_allie import AllieConfig
from chessnets.models.lc0 import AutoLc0Model
from chessnets.models.lc0.configuration_lc0 import (
    Lc0Config,
    Lc0ModelConfig,
    Lc0Policy,
)
from chessnets.models.maia2.configuration_maia2 import Config


def test_auto_lc0_model_dispatches_from_config(monkeypatch):
    config = SimpleNamespace(model=Lc0ModelConfig(policy=Lc0Policy.convolution))
    sentinel = object()
    monkeypatch.setattr(
        "chessnets.models.lc0.Lc0SeResNetModel.from_config",
        lambda received: sentinel,
    )

    assert AutoLc0Model.from_config(config) is sentinel


def test_model_configs_can_be_loaded_from_local_files(tmp_path):
    (tmp_path / "config.json").write_text(
        """{
          "model_type": "maia2",
          "config_schema_version": 1,
          "architectures": ["Maia2Model"],
          "time_control": "blitz",
          "model": {
            "input_channels": 18,
            "dim_cnn": 8,
            "dim_vit": 32,
            "num_blocks_cnn": 1,
            "num_blocks_vit": 1,
            "vit_length": 8,
            "elo_dim": 16
          }
        }"""
    )
    config = Config.from_pretrained(tmp_path)

    assert config.input_channels == 18
    assert config.dim_vit == 32
    assert config.time_control == "blitz"


def test_lc0_inference_config_can_be_loaded_without_training_fields(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n  filters: 8\n  residual_blocks: 1\n  se_ratio: 2\n"
    )
    config = Lc0Config.from_pretrained(tmp_path)

    assert config.model.filters == 8
    assert config.model.policy is Lc0Policy.convolution


def test_allie_config_preserves_nested_model_and_tokenizer_options():
    config = AllieConfig.from_dict(
        {
            "data_config": {"tokenizer_config": {"max_length": 128}},
            "model_config": {"base_model": "test-model"},
        }
    )

    assert config.data_config.tokenizer_config.max_length == 128
    assert config.model_config.base_model == "test-model"
