"""Lazy registry for published and externally registered chess pipelines.

Importing this module registers names only. Each factory imports its pipeline
(and PyTorch) when called, so looking up model IDs stays fast.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
import typing as tp

from chessnets.utils.time_control import TimeInfoPromptStyle


@dataclasses.dataclass(frozen=True)
class PipelineInfo:
    model_id: str
    factory_fn: tp.Callable[[tp.Any], tp.Any]

    def from_pretrained(self, device: tp.Any = None, **kwargs: tp.Any) -> tp.Any:
        if device is None:
            try:
                import torch
            except ImportError:
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.factory_fn(device, **kwargs)

    @functools.cached_property
    def path_friendly_model_id(self) -> str:
        return self.model_id.replace("/", "_").replace(" ", "_")


def create_lc0_pipeline(
    model_id: str,
    device: tp.Any,
    **kwargs: tp.Any,
) -> tp.Any:
    from .lc0.pipeline_lc0 import Lc0Pipeline

    return Lc0Pipeline.from_pretrained(model_id, device=device, **kwargs)


def create_unimaia_pipeline(
    model_id: str,
    device: tp.Any,
    pipeline_config: tp.Any = None,
    should_merge_lora_adapter: bool = False,
    dtype: tp.Any = None,
    **kwargs: tp.Any,
) -> tp.Any:
    from .unimaia.pipeline_unimaia import UniMaiaPipeline

    return UniMaiaPipeline.from_pretrained(
        model_id,
        device=device,
        dtype=dtype,
        should_merge_lora_adapter=should_merge_lora_adapter,
        pipeline_config=pipeline_config,
        **kwargs,
    )


def _published_unimaia_factory(
    model_id: str,
    default_mode: str,
    default_time_info_prompt_style: TimeInfoPromptStyle = TimeInfoPromptStyle.REWRITTEN,
):
    def factory(device: tp.Any, **kwargs: tp.Any):
        from .unimaia.pipeline_unimaia import UniMaiaInputMode, UniMaiaPipelineConfig

        input_mode = kwargs.pop("input_mode", None)
        pipeline_config = kwargs.pop(
            "pipeline_config",
            UniMaiaPipelineConfig(
                input_mode=UniMaiaInputMode(
                    default_mode if input_mode is None else input_mode
                ),
                time_info_prompt_style=default_time_info_prompt_style,
            ),
        )
        return create_unimaia_pipeline(
            model_id,
            device,
            pipeline_config=pipeline_config,
            **kwargs,
        )

    return factory


def create_chess_clip_pipeline(device: tp.Any, **kwargs: tp.Any) -> tp.Any:
    from .chess_clip.pipeline_chess_clip import ChessClipPipeline

    return ChessClipPipeline.from_pretrained(
        "Waterhorse/ChessCLIP", device=device, **kwargs
    )


def create_chessgpt_pipeline(
    checkpoint_id: str,
    pipeline_class_name: str,
    device: tp.Any,
    **kwargs: tp.Any,
) -> tp.Any:
    """Create one of the curated ChessGPT inference profiles lazily."""
    from .text_generation import pipeline_text_generation

    pipeline_class = getattr(pipeline_text_generation, pipeline_class_name)
    return pipeline_class.from_pretrained(
        checkpoint_id,
        device=device,
        **kwargs,
    )


PIPELINE_INFOS: dict[str, PipelineInfo] = {}
ARCHITECTURE_INFOS: dict[str, ArchitectureInfo] = {}


def _resolve_device(device: tp.Any) -> tp.Any:
    if device is not None:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclasses.dataclass(frozen=True)
class ArchitectureInfo:
    """Factory metadata for a model class named by ``config.json``."""

    architecture: str
    factory_fn: tp.Callable[..., tp.Any]

    def from_pretrained(
        self,
        model_id_or_path: str | pathlib.Path,
        device: tp.Any = None,
        **kwargs: tp.Any,
    ) -> tp.Any:
        """Call the factory with the source checkpoint and resolved device."""
        return self.factory_fn(
            model_id_or_path,
            _resolve_device(device),
            **kwargs,
        )


def _validate_architecture_name(name: str, *, kind: str = "architecture") -> str:
    if not isinstance(name, str) or not name or name.strip() != name:
        raise TypeError(f"{kind} name must be a non-empty string")
    if not name.isidentifier():
        raise ValueError(f"{kind} name must be a plain class name: {name!r}")
    return name


def _register_architecture(
    architecture: str,
    factory: tp.Callable[..., tp.Any],
    aliases: tp.Iterable[str] = (),
    *,
    overwrite: bool = False,
) -> None:
    architecture = _validate_architecture_name(architecture)
    if not callable(factory):
        raise TypeError("architecture factory must be callable")
    names = (architecture, *tuple(aliases))
    names = tuple(_validate_architecture_name(name) for name in names)
    if len(set(names)) != len(names):
        raise ValueError(
            f"Duplicate architecture registration name for {architecture!r}"
        )
    if not overwrite and any(name in ARCHITECTURE_INFOS for name in names):
        raise ValueError(f"An architecture is already registered for {architecture!r}")
    info = ArchitectureInfo(architecture=architecture, factory_fn=factory)
    for name in names:
        ARCHITECTURE_INFOS[name] = info


def _config_architectures(
    model_id_or_path: str | pathlib.Path,
    kwargs: dict[str, tp.Any],
) -> list[str]:
    """Read and validate static architecture metadata for a checkpoint."""
    from chessnets.models._pretrained import resolve_pretrained_file

    local_path = pathlib.Path(model_id_or_path).expanduser()
    if local_path.exists() and not local_path.is_dir():
        raise ValueError(
            f"A local model path must be a directory: {model_id_or_path!s}"
        )

    try:
        config_path = resolve_pretrained_file(
            model_id_or_path,
            "config.json",
            revision=kwargs.get("revision"),
            cache_dir=kwargs.get("cache_dir"),
            token=kwargs.get("token"),
            local_files_only=kwargs.get("local_files_only", False),
            repo_type=kwargs.get("repo_type"),
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"Unknown model_id {model_id_or_path!s}: no config.json was found"
        ) from exc

    try:
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read a valid config.json for {model_id_or_path!s}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError(
            f"config.json for {model_id_or_path!s} must contain a top-level JSON object"
        )
    architectures = metadata.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError(
            f"config.json for {model_id_or_path!s} must contain a non-empty "
            "'architectures' list"
        )
    if any(
        not isinstance(name, str) or not name or name.strip() != name
        for name in architectures
    ):
        raise ValueError(
            f"config.json for {model_id_or_path!s} has invalid "
            "'architectures' entries; they must be non-empty strings"
        )
    return architectures


class AutoPipeline:
    """Registry for built-in and external pipeline factories.

    Registration is metadata-only.  Model loaders and optional pipeline
    dependencies are imported when a pipeline is actually created.
    """

    @classmethod
    def register(
        cls,
        model_id: str,
        factory: tp.Callable[..., tp.Any],
        aliases: tp.Iterable[str] = (),
        *,
        overwrite: bool = False,
    ) -> None:
        names = (model_id, *tuple(aliases))
        if not overwrite and any(name in PIPELINE_INFOS for name in names):
            raise ValueError(f"A pipeline is already registered for {model_id!r}")
        for name in names:
            PIPELINE_INFOS[name] = PipelineInfo(model_id=model_id, factory_fn=factory)

    @classmethod
    def register_architecture(
        cls,
        architecture: str,
        factory: tp.Callable[..., tp.Any],
        aliases: tp.Iterable[str] = (),
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a factory selected by ``config.json`` metadata.

        ``factory`` is called as ``factory(model_id_or_path, device, **kwargs)``.
        Architecture names are static identifiers; they are never imported as
        Python modules and never execute code from a model repository.
        """
        _register_architecture(
            architecture,
            factory,
            aliases=aliases,
            overwrite=overwrite,
        )

    @classmethod
    def get_architecture_info(cls, architecture: str) -> ArchitectureInfo:
        try:
            return ARCHITECTURE_INFOS[architecture]
        except KeyError as exc:
            raise ValueError(f"Unknown model architecture: {architecture}") from exc

    @classmethod
    def get_pipeline_info(cls, model_id: str | pathlib.Path) -> PipelineInfo:
        try:
            return PIPELINE_INFOS[model_id]
        except KeyError as exc:
            raise ValueError(f"Unknown model_id: {model_id}") from exc

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | pathlib.Path,
        device: tp.Any = None,
        **kwargs: tp.Any,
    ) -> tp.Any:
        try:
            info = cls.get_pipeline_info(model_id)
        except ValueError as exact_id_error:
            architectures = _config_architectures(model_id, kwargs)
            for architecture in architectures:
                info = ARCHITECTURE_INFOS.get(architecture)
                if info is not None:
                    return info.from_pretrained(model_id, device=device, **kwargs)
            raise ValueError(
                f"Unknown model_id {model_id!s}: config.json architectures "
                f"{architectures!r} are not registered; register the model ID "
                "or one of these architecture names with AutoPipeline"
            ) from exact_id_error
        return info.from_pretrained(device=device, **kwargs)


# Kept as data here so importing AutoPipeline never imports the Lc0 loader and
# its protobuf/YAML/model implementation dependencies.
_LC0_MODEL_IDS = (
    "shermansiu/lc0_t3-512x15x16h-swa-2815000",
    "shermansiu/lc0_t82-768x15x24h-swa-7364000",
    "shermansiu/lc0_BT3-768x15x24h-swa-2790000",
    "shermansiu/lc0_BT4-1024x15x32h-swa-5000000",
    "shermansiu/lc0_BT4-1024x15x32h-swa-6147500",
    "shermansiu/lc0_BT4-spsa-1740",
    "shermansiu/maia-1100",
    "shermansiu/maia-1200",
    "shermansiu/maia-1300",
    "shermansiu/maia-1400",
    "shermansiu/maia-1500",
    "shermansiu/maia-1600",
    "shermansiu/maia-1700",
    "shermansiu/maia-1800",
    "shermansiu/maia-1900",
    "shermansiu/maia-2200",
)


def _register_builtins() -> None:
    for model_id in _LC0_MODEL_IDS:
        AutoPipeline.register(
            model_id,
            functools.partial(create_lc0_pipeline, model_id),
            aliases=(model_id.removeprefix("shermansiu/"),),
        )

    AutoPipeline.register(
        "shermansiu/unimaia",
        _published_unimaia_factory("shermansiu/unimaia", "prompt"),
        aliases=("unimaia",),
    )
    AutoPipeline.register(
        "shermansiu/unimaia-aux",
        _published_unimaia_factory(
            "shermansiu/unimaia-aux",
            "clocks",
            TimeInfoPromptStyle.LEGACY,
        ),
        aliases=("unimaia-aux",),
    )
    AutoPipeline.register("Waterhorse/ChessCLIP", create_chess_clip_pipeline)

    chessgpt_profiles = {
        "Waterhorse/chessgpt-chat-v1": (
            "Waterhorse/chessgpt-chat-v1",
            "HFTransformersProbabilityTextPipeline",
        ),
        "Waterhorse/chessgpt-chat-v1_simple": (
            "Waterhorse/chessgpt-chat-v1",
            "HFTransformersSimpleTextPipeline",
        ),
        "Waterhorse/chessgpt-chat-v1_simple_original": (
            "Waterhorse/chessgpt-chat-v1",
            "HFTransformersSimpleTextPipelineOriginal",
        ),
        "Waterhorse/chessgpt-chat-v1_general_policy": (
            "Waterhorse/chessgpt-chat-v1",
            "HFTransformersSimpleTextPipelineGeneralPolicy",
        ),
        "Waterhorse/chessgpt-base-v1": (
            "Waterhorse/chessgpt-base-v1",
            "HFTransformersSimpleTextPipelinePgnHeader",
        ),
        "Waterhorse/chessgpt-base-v1_with-time-control-and-elo": (
            "Waterhorse/chessgpt-base-v1",
            "HFTransformersSimpleTextPipelineCompletionWithTimeControlAndElo",
        ),
    }
    for profile_id, (checkpoint_id, pipeline_class_name) in chessgpt_profiles.items():
        AutoPipeline.register(
            profile_id,
            functools.partial(
                create_chessgpt_pipeline,
                checkpoint_id,
                pipeline_class_name,
            ),
        )
    AutoPipeline.register(
        "shermansiu/maia2-blitz",
        lambda device, **kwargs: _create_maia2("blitz", device, **kwargs),
        aliases=("maia2-blitz", "maia2_blitz"),
    )
    AutoPipeline.register(
        "shermansiu/maia2-rapid",
        lambda device, **kwargs: _create_maia2("rapid", device, **kwargs),
        aliases=("maia2-rapid", "maia2_rapid"),
    )
    AutoPipeline.register(
        "allie_medium_best",
        lambda device, **kwargs: _create_allie(device, **kwargs),
    )

    # These names are model class names, matching the Transformers-style
    # ``architectures`` metadata. Keep the mapping explicit and static.
    AutoPipeline.register_architecture("BTLeelaZeroNet", _create_lc0_from_architecture)
    AutoPipeline.register_architecture("SeLeelaZeroNet", _create_lc0_from_architecture)
    AutoPipeline.register_architecture(
        "UniMaiaModel", _create_unimaia_from_architecture
    )
    AutoPipeline.register_architecture("Maia2Model", _create_maia2_from_architecture)
    AutoPipeline.register_architecture(
        "CausalLMWithRegressionHead", _create_allie_from_architecture
    )


def _create_maia2(time_control: str, device: tp.Any, **kwargs: tp.Any):
    from .maia2.pipeline_maia2 import Maia2Pipeline

    return Maia2Pipeline.from_pretrained(
        **kwargs, time_control=time_control, device=device
    )


def _create_maia2_from_architecture(
    model_id: str | pathlib.Path, device: tp.Any, **kwargs: tp.Any
):
    from .maia2.pipeline_maia2 import Maia2Pipeline

    return Maia2Pipeline.from_pretrained(model_id, device=device, **kwargs)


def _create_allie(device: tp.Any, **kwargs: tp.Any):
    from .allie.pipeline_allie import AlliePipeline

    return AlliePipeline.from_pretrained(**kwargs, device=device)


def _create_allie_from_architecture(
    model_id: str | pathlib.Path, device: tp.Any, **kwargs: tp.Any
):
    from .allie.pipeline_allie import AlliePipeline

    return AlliePipeline.from_pretrained(model_id, device=device, **kwargs)


def _create_lc0_from_architecture(
    model_id: str | pathlib.Path, device: tp.Any, **kwargs: tp.Any
):
    return create_lc0_pipeline(model_id, device, **kwargs)


def _create_unimaia_from_architecture(
    model_id: str | pathlib.Path, device: tp.Any, **kwargs: tp.Any
):
    return create_unimaia_pipeline(model_id, device, **kwargs)


_register_builtins()

# Functional spelling for integrations that keep registries as module-level
# extension points, matching the classmethod's behavior.
register_architecture = AutoPipeline.register_architecture

__all__ = [
    "AutoPipeline",
    "PIPELINE_INFOS",
    "ARCHITECTURE_INFOS",
    "ArchitectureInfo",
    "register_architecture",
    "PipelineInfo",
    "create_chess_clip_pipeline",
    "create_chessgpt_pipeline",
    "create_lc0_pipeline",
    "create_unimaia_pipeline",
]
