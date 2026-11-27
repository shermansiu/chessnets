"""Configuration for Lc0 and Maia-1 networks.

``Lc0InferenceConfig`` is the ``config.json`` shipped with published checkpoints;
``Lc0Config`` is the full lczero-training YAML configuration.
"""

import bisect
import dataclasses
import enum
import json
import os
import pathlib
import typing as tp

import dacite
import torch
import yaml


class AutoStrEnum(str, enum.Enum):
    """
    StrEnum where enum.auto() returns the field name.
    See https://docs.python.org/3.9/library/enum.html#using-automatic-values
    """

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list
    ) -> str:
        return name


class Lc0Policy(AutoStrEnum):
    classical = enum.auto()
    convolution = enum.auto()
    attention = enum.auto()


class Lc0Value(AutoStrEnum):
    classical = enum.auto()
    wdl = enum.auto()

    @property
    def is_wdl(self):
        return self == Lc0Value.wdl


class Lc0Input(AutoStrEnum):
    classic = enum.auto()
    frc_castling = enum.auto()
    canonical = enum.auto()
    canonical_100 = enum.auto()
    canonical_armageddon = enum.auto()
    canonical_v2 = enum.auto()
    canonical_v2_armageddon = enum.auto()


def identity_function(value):
    return value


class Lc0Activation(AutoStrEnum):
    """Leela network non-linear activations.

    See https://www.tensorflow.org/api_docs/python/tf/keras/activations
    """

    elu = enum.auto()
    exponential = enum.auto()
    gelu = enum.auto()
    hard_sigmoid = enum.auto()
    linear = enum.auto()
    mish = enum.auto()
    relu = enum.auto()
    selu = enum.auto()
    sigmoid = enum.auto()
    softmax = enum.auto()
    softsign = enum.auto()
    swish = enum.auto()
    tanh = enum.auto()

    @property
    def as_fn(self):
        match self:
            case Lc0Activation.elu:
                return torch.nn.functional.elu
            case Lc0Activation.exponential:
                return torch.exp
            case Lc0Activation.gelu:
                return torch.nn.functional.gelu
            case Lc0Activation.hard_sigmoid:
                return torch.nn.functional.hardsigmoid
            case Lc0Activation.linear:
                return identity_function
            case Lc0Activation.mish:
                return torch.nn.functional.mish
            case Lc0Activation.relu:
                return torch.nn.functional.relu
            case Lc0Activation.selu:
                return torch.nn.functional.selu
            case Lc0Activation.sigmoid:
                return torch.nn.functional.sigmoid
            case Lc0Activation.softmax:
                return torch.nn.functional.softmax
            case Lc0Activation.softsign:
                return torch.nn.functional.softsign
            case Lc0Activation.swish:
                return torch.nn.functional.silu
            case Lc0Activation.tanh:
                return torch.nn.functional.tanh
            case _:
                return identity_function


class Lc0EmbeddingStyle(AutoStrEnum):
    new = enum.auto()
    old = enum.auto()


class Lc0MovesLeft(AutoStrEnum):
    none = enum.auto()
    v1 = enum.auto()

    @property
    def is_enabled(self):
        return self == Lc0MovesLeft.v1


@dataclasses.dataclass(frozen=True)
class Lc0ModelConfig:
    arc_encoding: bool = False
    categorical_value_buckets: int = 0
    default_activation: Lc0Activation | None = Lc0Activation.mish
    dropout_rate: float = 0.0
    embedding_dense_sz: int = 128
    embedding_size: int = 768
    embedding_style: Lc0EmbeddingStyle = Lc0EmbeddingStyle.new
    encoder_layers: int = 15
    encoder_heads: int = 24
    encoder_d_model: int = 768
    encoder_dff: int = 1024
    ffn_activation: Lc0Activation | None = Lc0Activation.mish  # None means linear
    filters: int = 64
    input_type: Lc0Input = Lc0Input.classic
    moves_left: Lc0MovesLeft = Lc0MovesLeft.none
    moves_left_embedding_size: int = 8
    omit_qkv_biases: bool = False
    policy: Lc0Policy = Lc0Policy.convolution
    policy_channels: int = 32
    policy_d_aux: int = 768
    policy_d_model: int = 768
    policy_embedding_size: int = 768
    policy_next: bool = False
    policy_opponent: bool = False
    policy_optimistic_st: bool = False
    residual_blocks: int = 6
    se_ratio: int = 8
    skip_first_ln: bool = False
    smolgen_activation: Lc0Activation | None = Lc0Activation.linear  # None means linear
    smolgen_gen_sz: int = 256
    smolgen_hidden_channels: int = 32
    smolgen_hidden_sz: int = 256
    soft_policy: bool = False
    soft_policy_temperature: float = 4.0
    use_smolgen: bool = True
    value: Lc0Value = Lc0Value.wdl
    value_st_type: Lc0Value = (
        Lc0Value.wdl
    )  # Not in original. Added for the ability to fully load BT3 weights.
    value_st_cat: bool = (
        True  # Not in original. Added for the ability to fully load BT3 weights.
    )
    value_embedding_size: int = 32
    value_q: bool = False
    value_q_cat: bool = (
        True  # Not in original. Added for the ability to fully load T3 weights.
    )
    value_st: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, tp.Any]) -> "Lc0ModelConfig":
        return dacite.from_dict(
            data_class=cls,
            data=dict(values),
            config=dacite.Config(cast=[enum.Enum]),
        )

    @property
    def default_activation_fn(self):
        if self.default_activation is None:
            return identity_function
        return self.default_activation.as_fn

    @property
    def ffn_activation_fn(self):
        if self.ffn_activation is None:
            return identity_function
        return self.ffn_activation.as_fn

    @property
    def smolgen_activation_fn(self):
        if self.smolgen_activation is None:
            return identity_function
        return self.smolgen_activation.as_fn


def game_number_for_name(name):
    num_str = os.path.basename(name).upper().strip("ABCDEFGHIJKLMNOPQRSTUVWXYZ_-.")
    return int(num_str)


class Lc0SortType(AutoStrEnum):
    mtime = enum.auto()
    number = enum.auto()
    name = enum.auto()

    def fn(self):
        match self:
            case Lc0SortType.mtime:
                return os.path.getmtime
            case Lc0SortType.number:
                return game_number_for_name
            case Lc0SortType.name:
                return identity_function
            case _:
                raise ValueError(f"Unknown dataset sort_type: {self}")


def optional_pathlist(
    pathlist: pathlib.Path | list[pathlib.Path] | None,
) -> list[pathlib.Path] | None:
    if pathlist is None:
        return None
    if isinstance(pathlist, (str, bytes, pathlib.Path)):
        return [pathlist]
    return pathlist


@dataclasses.dataclass(frozen=True)
class Lc0DatasetConfig:
    num_chunks: int
    train_ratio: float
    allow_less_chunks: bool = False
    fast_chunk_loading = True
    sort_type: Lc0SortType = Lc0SortType.mtime
    input_train: list[pathlib.Path] | None = None
    input_validation: list[pathlib.Path] | None = None
    input_test: list[pathlib.Path] | None = None
    input: list[pathlib.Path] | None = None
    train_workers: int | None = None
    test_workers: int | None = None
    experimental_v5_only_dataset: bool = False

    def __post_init__(self) -> None:
        for field_name in ("input_train", "input_validation", "input_test", "input"):
            paths = optional_pathlist(getattr(self, field_name))
            if paths is not None:
                paths = [pathlib.Path(path) for path in paths]
            object.__setattr__(self, field_name, paths)


class Lc0Precision(AutoStrEnum):
    single = enum.auto()
    half = enum.auto()

    @property
    def dtype(self) -> torch.dtype:
        match self:
            case Lc0Precision.single:
                return torch.float32
            case Lc0Precision.half:
                return torch.float16
            case _:
                raise ValueError(f"Unknown precision: {self}")


class Lc0Optimizer(AutoStrEnum):
    sgd = enum.auto()
    rmsprop = enum.auto()
    nadam = enum.auto()
    adam = enum.auto()
    adabelief = enum.auto()


@dataclasses.dataclass(frozen=True)
class Lc0LossWeightsConfig:
    policy: float = 1.0
    policy_optimistic_st: float = 4.0
    policy_soft: float = 4.0  # loss is much smaller at optimum
    value_winner: float = 1.0
    value_q: float = 1.0
    value_st: float = 5.0  # larger because mse loss
    value_q_err: float = 5.0  # both error weights should be the same
    value_st_err: float = 5.0
    moves_left: float = 1.0
    reg: float = 1.0


@dataclasses.dataclass(frozen=True)
class Lc0TrainingConfig:
    batch_size: int
    path: pathlib.Path
    shuffle_size: int
    test_steps: int
    total_steps: int
    train_avg_report_steps: int
    accuracy_thresholds: list[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 5, 10]
    )
    beta1: float = 0.9  # Nadam beta1 default is 0.9
    beta2: float = 0.999  # Nadam beta2 default is 0.999
    checkpoint_steps: int | None = None
    diff_focus_min: float = 1.0
    diff_focus_slope: float = 0.0
    diff_focus_q_weight: float = 6.0
    diff_focus_pol_scale: float = 3.5
    epsilon: float = 1e-07  # Nadam epsilon value
    lookahead_optimzier: bool = False
    loss_scale: int = 128
    loss_weights: Lc0LossWeightsConfig = dataclasses.field(
        default_factory=Lc0LossWeightsConfig
    )
    lr_boundaries: list[int] = dataclasses.field(default_factory=list)
    lr_values: list[float] = dataclasses.field(default_factory=list)
    mask_legal_moves: bool = False
    max_grad_norm: float = 10000.0
    num_batch_splits: int = 1
    num_test_positions: int = 131_072
    optimizer: Lc0Optimizer = Lc0Optimizer.sgd
    precision: Lc0Precision = Lc0Precision.single
    q_ratio: int = 0
    renorm: bool = False
    renorm_max_r: float = 1.0
    renorm_max_d: float = 0.0
    renorm_momentum: float = 0.99
    sparse: bool = False
    swa: bool = False
    swa_max_n: int = (
        0  # Limit momentum of SWA exponential average to 1 - 1/(swa_max_n + 1)
    )
    swa_output: bool = False
    validation_steps: int = 2500
    value_focus_min: float = 0.025
    value_focus_slope: float = 3.0
    virtual_batch_size: int | None = None
    warmup_steps: int = 0
    weight_decay: float = 0.0
    policy_loss_weight: float = 1.0  # Legacy loss weight config
    value_loss_weight: float = 1.0
    moves_left_loss_weight: float = 1.0

    def __post_init__(self):
        self.lr_boundaries.sort()

    @property
    def swa_enabled(self) -> bool:
        return self.swa

    @property
    def renorm_enabled(self) -> bool:
        return self.renorm

    def lr(self, steps: int) -> float:
        steps_total = steps % self.total_steps
        lr = self.lr_values[bisect.bisect_right(self.lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            lr = lr * (steps + 1) / self.warmup_steps
        return lr

    @property
    def optimizer_name(self):
        return self.optimizer


@dataclasses.dataclass(frozen=True)
class Lc0OptionsConfig:
    nodes: int
    weightsPath: pathlib.Path
    movetime: int
    threads: int


@dataclasses.dataclass(frozen=True)
class Lc0Config:
    model: Lc0ModelConfig
    training: Lc0TrainingConfig
    name: str
    engine: str | None = None
    options: Lc0OptionsConfig | None = None
    dataset: Lc0DatasetConfig | None = None

    gpu: str | int = 0
    display_name: str | None = None

    @property
    def dtype(self):
        return self.training.precision.dtype

    @property
    def root_dir(self) -> pathlib.Path:
        return self.training.path / self.name

    @classmethod
    def from_dict(cls, values: dict[str, tp.Any]) -> "Lc0Config":
        """Build a config from plain YAML-compatible values."""
        values = dict(values)
        if values.get("dataset") is not None:
            dataset = dict(values["dataset"])
            for field_name in (
                "input_train",
                "input_validation",
                "input_test",
                "input",
            ):
                if field_name in dataset:
                    dataset[field_name] = optional_pathlist(dataset[field_name])
            values["dataset"] = dataset
        return dacite.from_dict(
            data_class=cls,
            data=values,
            config=dacite.Config(
                cast=[enum.Enum],
                type_hooks={pathlib.Path: pathlib.Path},
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "Lc0Config":
        with pathlib.Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path,
        *,
        revision: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        repo_type: str | None = None,
    ) -> "Lc0Config | Lc0InferenceConfig":
        """Load the architecture config from a checkpoint ID or directory."""
        from chessnets.models._pretrained import resolve_pretrained_file

        resolve_kwargs = {
            "revision": revision,
            "cache_dir": cache_dir,
            "token": token,
            "local_files_only": local_files_only,
            "repo_type": repo_type,
        }
        try:
            path = resolve_pretrained_file(
                pretrained_model_name_or_path, "config.json", **resolve_kwargs
            )
        except FileNotFoundError:
            path = resolve_pretrained_file(
                pretrained_model_name_or_path, "config.yaml", **resolve_kwargs
            )
        with path.open(encoding="utf-8") as stream:
            if path.suffix == ".json":
                values = json.load(stream)
            else:
                values = yaml.safe_load(stream) or {}
        if "training" not in values or "name" not in values:
            return Lc0InferenceConfig.from_dict(values)
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, tp.Any]:
        """Return plain values suitable for YAML safe serialization."""

        def plain(value: tp.Any) -> tp.Any:
            if dataclasses.is_dataclass(value):
                return {
                    field.name: plain(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                }
            if isinstance(value, enum.Enum):
                return value.value
            if isinstance(value, pathlib.Path):
                return str(value)
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            return value

        return tp.cast(dict[str, tp.Any], plain(self))


@dataclasses.dataclass(frozen=True)
class Lc0InferenceConfig:
    """The model-only subset shipped with published Lc0 checkpoints."""

    model: Lc0ModelConfig
    model_type: str = "lc0"
    config_schema_version: int = 1
    architectures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.model_type != "lc0":
            raise ValueError(f"Unsupported Lc0 model_type: {self.model_type!r}")
        if self.config_schema_version != 1:
            raise ValueError(
                f"Unsupported Lc0 config_schema_version: {self.config_schema_version}"
            )
        if not self.architectures:
            architecture = (
                "Lc0ChessFormerModel"
                if self.model.policy is Lc0Policy.attention
                else "Lc0SeResNetModel"
            )
            object.__setattr__(self, "architectures", (architecture,))

    @classmethod
    def from_dict(cls, values: dict[str, tp.Any]) -> "Lc0InferenceConfig":
        if not isinstance(values, dict):
            raise TypeError(f"Expected a mapping, got {type(values).__name__}")
        if "model" not in values:
            raise ValueError("Lc0 config must contain a top-level 'model' mapping")
        return cls(
            model=Lc0ModelConfig.from_dict(values["model"]),
            model_type=values.get("model_type", "lc0"),
            config_schema_version=values.get("config_schema_version", 1),
            architectures=tuple(values.get("architectures", ())),
        )

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "Lc0InferenceConfig":
        with pathlib.Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "Lc0InferenceConfig":
        with pathlib.Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})

    def to_dict(self) -> dict[str, tp.Any]:
        def plain(value: tp.Any) -> tp.Any:
            if dataclasses.is_dataclass(value):
                return {
                    field.name: plain(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                }
            if isinstance(value, enum.Enum):
                return value.value
            if isinstance(value, tuple):
                return [plain(item) for item in value]
            return value

        return {
            "model_type": self.model_type,
            "config_schema_version": self.config_schema_version,
            "architectures": list(self.architectures),
            "model": plain(self.model),
        }

    def save_pretrained(self, directory: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(directory) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path
