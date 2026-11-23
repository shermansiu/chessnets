import dataclasses
import pathlib

import yaml

from chessnets.models.lc0_senet.configuration_lc0_senet import (
    LeelaActivation,
    LeelaInferenceConfig,
    LeelaNetConfig,
    LeelaNetModelConfig,
    LeelaOptimizer,
    LeelaPrecision,
)


def test_inference_config_json_materializes_model_defaults(tmp_path):
    config = LeelaInferenceConfig(model=LeelaNetModelConfig(filters=32))
    path = config.save_pretrained(tmp_path)

    values = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert values["model_type"] == "lc0"
    assert values["config_schema_version"] == 1
    assert values["model"]["filters"] == 32
    assert values["model"]["encoder_layers"] == 15
    assert LeelaInferenceConfig.from_json(path) == config


def test_nested_config_yaml_types_and_round_trip(tmp_path):
    values = {
        "name": "test-net",
        "model": {"default_activation": "relu", "filters": 128},
        "training": {
            "batch_size": 32,
            "path": str(tmp_path / "runs"),
            "shuffle_size": 100,
            "test_steps": 10,
            "total_steps": 1000,
            "train_avg_report_steps": 5,
            "optimizer": "adam",
            "precision": "half",
            "diff_focus_min": 0.025,
            "diff_focus_slope": 3.0,
            "renorm_max_r": 1.0,
            "renorm_max_d": 0.0,
            "lr_boundaries": [500, 100],
            "lr_values": [0.1, 0.01, 0.001],
            "loss_weights": {"policy": 2.5},
        },
        "dataset": {
            "num_chunks": 10,
            "train_ratio": 0.9,
            "input_train": "data/train",
            "input_validation": ["data/validation"],
        },
        "options": {
            "nodes": 100,
            "weightsPath": "weights/model.pb.gz",
            "movetime": 1000,
            "threads": 2,
        },
        "gpu": "cuda:0",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    config = LeelaNetConfig.from_yaml(path)

    assert dataclasses.is_dataclass(config)
    assert config.model.default_activation is LeelaActivation.relu
    assert config.training.optimizer is LeelaOptimizer.adam
    assert config.training.precision is LeelaPrecision.half
    assert config.training.path == tmp_path / "runs"
    assert config.training.loss_weights.policy == 2.5
    assert config.training.diff_focus_min == 0.025
    assert config.training.diff_focus_slope == 3.0
    assert config.training.renorm_max_r == 1.0
    assert config.training.renorm_max_d == 0.0
    assert config.training.lr_boundaries == [100, 500]
    assert config.dataset.input_train == [pathlib.Path("data/train")]
    assert config.options.weightsPath == pathlib.Path("weights/model.pb.gz")
    assert config.gpu == "cuda:0"

    serialized = config.to_dict()
    yaml.safe_dump(serialized)
    restored = LeelaNetConfig.from_dict(serialized)
    assert restored == config


def test_gpu_preserves_str_or_int(tmp_path):
    values = {
        "name": "test-net",
        "model": {},
        "training": {
            "batch_size": 1,
            "path": str(tmp_path),
            "shuffle_size": 1,
            "test_steps": 1,
            "total_steps": 1,
            "train_avg_report_steps": 1,
        },
    }

    assert LeelaNetConfig.from_dict({**values, "gpu": 0}).gpu == 0
    assert LeelaNetConfig.from_dict({**values, "gpu": "0"}).gpu == "0"
