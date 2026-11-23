"""Strict loading helpers for single-file and sharded safetensors checkpoints."""

from __future__ import annotations

import json
import pathlib
import typing as tp

import torch

from chessnets.models._pretrained import resolve_pretrained_file


def _safe_shard_name(name: str) -> str:
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.name != name:
        raise ValueError(f"Unsafe safetensors shard name: {name!r}")
    if not name.endswith(".safetensors"):
        raise ValueError(f"Invalid safetensors shard name: {name!r}")
    return name


def load_safetensors_file(
    checkpoint_path: str | pathlib.Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a local single-file or indexed safetensors checkpoint."""
    from safetensors.torch import load_file

    path = pathlib.Path(checkpoint_path)
    if path.name == "model.safetensors.index.json" or path.name.endswith(
        ".safetensors.index.json"
    ):
        index = json.loads(path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors index in {path}")
        expected_by_shard: dict[str, set[str]] = {}
        for key, shard in weight_map.items():
            if not isinstance(key, str) or not isinstance(shard, str):
                raise ValueError(f"Invalid weight_map entry in {path}")
            expected_by_shard.setdefault(_safe_shard_name(shard), set()).add(key)

        state_dict: dict[str, torch.Tensor] = {}
        for shard, expected_keys in expected_by_shard.items():
            shard_path = path.parent / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
            shard_state = load_file(str(shard_path), device=str(device))
            missing = expected_keys.difference(shard_state)
            if missing:
                raise ValueError(
                    f"Shard {shard!r} is missing indexed tensors: {sorted(missing)}"
                )
            for key in expected_keys:
                if key in state_dict:
                    raise ValueError(f"Duplicate tensor {key!r} in {path}")
                state_dict[key] = shard_state[key]
        return state_dict

    if path.suffix != ".safetensors":
        raise ValueError(f"Unsupported safetensors checkpoint: {path.name}")
    return load_file(str(path), device=str(device))


ModelT = tp.TypeVar("ModelT", bound=torch.nn.Module)


def build_with_state_dict(
    build: tp.Callable[[], ModelT],
    state_dict: tp.Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> ModelT:
    """Build a model and adopt ``state_dict`` without random initialization.

    The model is constructed on the ``meta`` device, so no parameter memory is
    allocated or initialized, and the checkpoint tensors are then assigned in
    place of the meta parameters. This skips the weight initialization that a
    normal construction performs only for the checkpoint to overwrite it.

    Non-persistent buffers (for example policy index maps) are not in the
    checkpoint, so they must be created from non-factory data such as
    ``torch.from_numpy``, which the meta device does not intercept. Anything
    still on ``meta`` after loading is reported as an error.
    """
    with torch.device("meta"):
        model = build()
    model.load_state_dict(state_dict, strict=strict, assign=True)
    unmaterialized = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.is_meta
    ]
    if unmaterialized:
        raise RuntimeError(
            "Tensors were not provided by the checkpoint and cannot be built on "
            f"the meta device: {unmaterialized}"
        )
    return model


def resolve_safetensors_checkpoint(
    model_id_or_path: str | pathlib.Path,
    **resolve_kwargs: tp.Any,
) -> pathlib.Path:
    """Resolve an indexed checkpoint first, then a single-file checkpoint."""
    path = pathlib.Path(model_id_or_path).expanduser()
    if path.is_file():
        if path.suffix == ".safetensors" or path.name.endswith(
            ".safetensors.index.json"
        ):
            return path.resolve()
        raise FileNotFoundError(f"Not a safetensors checkpoint: {path}")

    for filename in ("model.safetensors.index.json", "model.safetensors"):
        try:
            resolved = resolve_pretrained_file(
                model_id_or_path, filename, **resolve_kwargs
            )
        except FileNotFoundError:
            continue
        if filename.endswith(".index.json") and not path.is_dir():
            index = json.loads(resolved.read_text(encoding="utf-8"))
            for shard in set(index.get("weight_map", {}).values()):
                _safe_shard_name(shard)
                resolve_pretrained_file(model_id_or_path, shard, **resolve_kwargs)
        return resolved
    raise FileNotFoundError(
        f"No model.safetensors checkpoint was found in {model_id_or_path!s}"
    )


__all__ = [
    "build_with_state_dict",
    "load_safetensors_file",
    "resolve_safetensors_checkpoint",
]
