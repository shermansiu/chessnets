"""Load UniMaia checkpoints from Hugging Face or local directories."""

from __future__ import annotations

import json
import pathlib
import typing as tp

import numpy as np
import torch
from huggingface_hub.utils import EntryNotFoundError

from chessnets._optional import optional_module
from chessnets.loaders.safetensors import (
    load_safetensors_file,
    resolve_safetensors_checkpoint,
)
from chessnets.models.lc0_senet import configuration_lc0_senet as leela_config
from chessnets.models.unimaia.configuration_unimaia import (
    TextEncoderType,
    UniMaiaConfig,
)

accelerate = optional_module("accelerate", "unimaia")
transformers = optional_module("transformers", "unimaia")

PUBLISHED_UNIMAIA_MODELS = {
    "shermansiu/unimaia",
    "shermansiu/unimaia-aux",
}


def _is_local(value: str | pathlib.Path) -> bool:
    return pathlib.Path(value).expanduser().exists()


def _resolve_artifact(
    model_id_or_path: str | pathlib.Path,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | pathlib.Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    repo_type: str | None = None,
) -> pathlib.Path:
    path = pathlib.Path(model_id_or_path).expanduser()
    if path.exists():
        if path.is_file():
            return path
        candidate = path / filename
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find {filename!r} in {path}")
        return candidate

    from chessnets.models._pretrained import resolve_pretrained_file

    return resolve_pretrained_file(
        model_id_or_path,
        filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        repo_type=repo_type,
    )


def _resolve_config_path(
    model_id_or_path: str | pathlib.Path,
    config: UniMaiaConfig | None,
) -> pathlib.Path:
    if config is not None and config.model_path:
        return pathlib.Path(config.model_path).expanduser()
    return _resolve_artifact(model_id_or_path, "config.yaml")


def _remap_lora_state_dict(
    state_dict: dict[str, torch.Tensor], target_lora_modules: list[str]
) -> dict[str, torch.Tensor]:
    def remap_prefix(name: str) -> str:
        if name.startswith("text_encoder") and not name.startswith(
            "text_encoder.base_model.model"
        ):
            return "text_encoder.base_model.model" + name.removeprefix("text_encoder")
        return name

    def remap_suffix(name: str) -> str:
        if not name.startswith("text_encoder"):
            return name
        if name.endswith("weight") and not name.endswith(
            ("default.weight", "base_layer.weight")
        ):
            if "lora_A" in name or "lora_B" in name:
                return name.removesuffix("weight") + "default.weight"
            if any(module in name for module in target_lora_modules):
                return name.removesuffix("weight") + "base_layer.weight"
        if name.endswith("bias") and not name.endswith("base_layer.bias"):
            if any(module in name for module in target_lora_modules):
                return name.removesuffix("bias") + "base_layer.bias"
        return name

    return {
        remap_suffix(remap_prefix(name)): value for name, value in state_dict.items()
    }


def _remap_old_t5gemma2_state_dict(state_dict: dict[str, torch.Tensor]) -> dict:
    return {
        name.replace(
            "text_encoder.base_model.model.layers.",
            "text_encoder.base_model.model.text_model.layers.",
        )
        .replace(
            "text_encoder.base_model.model.embed_tokens.",
            "text_encoder.base_model.model.text_model.embed_tokens.",
        )
        .replace(
            "text_encoder.base_model.model.norm.",
            "text_encoder.base_model.model.text_model.norm.",
        )
        .replace(
            "text_encoder.base_model.model.vision_tower.vision_model.",
            "text_encoder.base_model.model.vision_tower.",
        ): value
        for name, value in state_dict.items()
    }


def _load_local_training_checkpoint(
    checkpoint_path: str | pathlib.Path,
) -> dict[str, torch.Tensor]:
    """Read model tensors from one trusted local UniMaia training checkpoint."""
    path = pathlib.Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Local training checkpoint does not exist: {path}")
    if path.suffix != ".pt":
        raise ValueError(f"Expected a local .pt training checkpoint: {path}")

    reconstruct = (
        np.core.multiarray._reconstruct
        if np.__version__.startswith("1.")
        else np._core.multiarray._reconstruct
    )
    safe_globals: list[tp.Any] = [
        reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    if not np.__version__.startswith("1."):
        safe_globals.append((reconstruct, "numpy.core.multiarray._reconstruct"))
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    if not isinstance(checkpoint, dict):
        raise TypeError("UniMaia training checkpoint must contain a mapping")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError(
            "UniMaia training checkpoint must contain a non-empty 'model' mapping"
        )
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise TypeError(
            "UniMaia training checkpoint 'model' must map strings to tensors"
        )
    return state_dict


def load_unimaia_model(
    model_id_or_path: str | pathlib.Path,
    *,
    config: UniMaiaConfig | None = None,
    checkpoint_format: tp.Literal["safetensors", "training_pt"] = "safetensors",
    is_eval: bool = True,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    should_merge_lora_adapter: bool = False,
    revision: str | None = None,
    cache_dir: str | pathlib.Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    repo_type: str | None = None,
) -> tuple[torch.nn.Module, tp.Any]:
    """Load a UniMaia model and tokenizer from a repository or local path."""
    from chessnets.models.unimaia import modeling_unimaia as unimaia_model

    artifact_kwargs = {
        "revision": revision,
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
        "repo_type": repo_type,
    }
    if checkpoint_format not in {"safetensors", "training_pt"}:
        raise ValueError(
            f"Unsupported UniMaia checkpoint format: {checkpoint_format!r}"
        )
    if checkpoint_format == "training_pt":
        local_checkpoint = pathlib.Path(model_id_or_path).expanduser()
        if config is None:
            raise ValueError(
                "A resolved UniMaia config is required for a training_pt checkpoint"
            )
        if not local_checkpoint.is_file():
            raise ValueError("training_pt checkpoints must be existing local files")
    elif config is None:
        try:
            config_path = _resolve_artifact(
                model_id_or_path, "config.json", **artifact_kwargs
            )
        except FileNotFoundError:
            config = UniMaiaConfig.from_yaml(
                _resolve_artifact(model_id_or_path, "config.yaml", **artifact_kwargs)
            )
        else:
            config_values = json.loads(config_path.read_text(encoding="utf-8"))
            has_runtime_fields = any(
                name in config_values
                for name in ("lc0_model_id", "lc0_bt_model_id", "base_lc0_model_id")
            )
            if has_runtime_fields:
                config = UniMaiaConfig.from_dict(config_values)
            else:
                # Transitional model repos used config.json only for AutoPipeline
                # dispatch and kept inference fields in config.yaml.
                config = UniMaiaConfig.from_yaml(
                    _resolve_artifact(
                        model_id_or_path, "config.yaml", **artifact_kwargs
                    )
                )
    if device is None:
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = config.model_dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if isinstance(dtype, str):
        try:
            dtype = dtype_map[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported UniMaia dtype: {dtype!r}") from exc

    local_root = pathlib.Path(model_id_or_path).expanduser()
    tokenizer_source = config.tokenizer_path
    if tokenizer_source is None:
        tokenizer_source = (
            str(local_root) if local_root.is_dir() else config.llm_model_id
        )
    elif local_root.is_dir() and not pathlib.Path(tokenizer_source).is_absolute():
        tokenizer_source = str(local_root / tokenizer_source)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lc0_config_path = config.lc0_config_path
    if config.lc0_config is not None:
        lc0_config = leela_config.LeelaInferenceConfig.from_dict(config.lc0_config)
    elif lc0_config_path is None:
        lc0_config = leela_config.LeelaNetConfig.from_pretrained(
            config.lc0_model_id, **artifact_kwargs
        )
    else:
        if local_root.is_dir() and not pathlib.Path(lc0_config_path).is_absolute():
            lc0_config_path = str(local_root / lc0_config_path)
        lc0_path = pathlib.Path(lc0_config_path)
        lc0_config = (
            leela_config.LeelaInferenceConfig.from_json(lc0_path)
            if lc0_path.suffix == ".json"
            else leela_config.LeelaInferenceConfig.from_yaml(lc0_path)
        )

    text_config_source = config.llm_config_path or config.llm_model_id
    if (
        config.llm_config_path
        and local_root.is_dir()
        and not pathlib.Path(text_config_source).is_absolute()
    ):
        text_config_source = str(local_root / text_config_source)
    text_config = transformers.AutoConfig.from_pretrained(
        text_config_source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    with accelerate.init_empty_weights():
        if config.llm_model_id.startswith("google/gemma-3-"):
            text_model = transformers.AutoModelForCausalLM.from_config(
                text_config, dtype=dtype
            ).model
        else:
            text_model = transformers.AutoModel.from_config(text_config, dtype=dtype)
            if config.llm_model_id.startswith("google/t5gemma-2-"):
                text_model = text_model.encoder

        model = unimaia_model.UniMaiaModel.from_config(
            config=lc0_config,
            umt_config=config,
            text_encoder=text_model,
        )

    checkpoint_path = (
        str(local_checkpoint)
        if checkpoint_format == "training_pt"
        else config.model_path
    )
    if (
        checkpoint_path is not None
        and local_root.is_dir()
        and not pathlib.Path(checkpoint_path).is_absolute()
    ):
        checkpoint_path = str(local_root / checkpoint_path)
    if checkpoint_path is None:
        try:
            checkpoint_path = str(
                resolve_safetensors_checkpoint(model_id_or_path, **artifact_kwargs)
            )
        except (FileNotFoundError, EntryNotFoundError):
            pass
    if checkpoint_path is None:
        raise FileNotFoundError("No UniMaia model checkpoint was found")

    if checkpoint_format == "training_pt":
        state_dict = _load_local_training_checkpoint(checkpoint_path)
    else:
        state_dict = load_safetensors_file(
            pathlib.Path(checkpoint_path), device=device or "cpu"
        )
    if config.llm_model_id.startswith("google/t5gemma-2-"):
        state_dict = _remap_old_t5gemma2_state_dict(state_dict)
    if config.text_encoder_type == TextEncoderType.lora:
        state_dict = _remap_lora_state_dict(state_dict, config.target_lora_modules)
        model.load_state_dict(state_dict, strict=False, assign=True)
    else:
        model.load_state_dict(state_dict, assign=True)

    if should_merge_lora_adapter and hasattr(model.text_encoder, "merge_and_unload"):
        model.text_encoder = model.text_encoder.merge_and_unload()
    model.to(device=device, dtype=dtype)
    if is_eval:
        model.eval()
    return model, tokenizer


__all__ = ["PUBLISHED_UNIMAIA_MODELS", "load_unimaia_model"]
