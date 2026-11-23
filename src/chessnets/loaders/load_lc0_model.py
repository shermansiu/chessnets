"""Safetensors-first loading for Lc0 and Maia-1 checkpoints."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import pathlib
import typing as tp

import torch
from chessnets.loaders.safetensors import (
    build_with_state_dict,
    load_safetensors_file,
    resolve_safetensors_checkpoint,
)
from chessnets.models.lc0_cf import modeling_lc0_cf
from chessnets.models.lc0_senet import configuration_lc0_senet as leela_config
from chessnets.models.lc0_senet import modeling_lc0_senet

Lc0Model = modeling_lc0_cf.BTLeelaZeroNet | modeling_lc0_senet.SeLeelaZeroNet


def _model_from_config(
    config: leela_config.LeelaNetConfig | leela_config.LeelaInferenceConfig,
) -> Lc0Model:
    if config.model.policy is leela_config.LeelaPolicy.attention:
        return modeling_lc0_cf.BTLeelaZeroNet.from_config(config)
    if config.model.policy is leela_config.LeelaPolicy.convolution:
        return modeling_lc0_senet.SeLeelaZeroNet.from_config(config)
    raise ValueError(f"Unsupported Lc0 policy architecture: {config.model.policy!r}")


def _set_mode(model: Lc0Model, is_eval: bool) -> Lc0Model:
    if is_eval:
        model.eval()
    return model


def load_pretrained_model(
    model_id: str | pathlib.Path,
    is_eval: bool = True,
    *,
    checkpoint_format: tp.Literal["auto", "safetensors", "protobuf"] = "auto",
    use_safetensors: bool | None = None,
    device: str | torch.device = "cpu",
    **kwargs: tp.Any,
) -> Lc0Model:
    """Load Lc0/Maia weights, preferring safetensors in automatic mode.

    ``checkpoint_format`` is a loader choice and is deliberately not stored in
    the architecture config. ``use_safetensors`` mirrors the Transformers API
    and is retained as a convenience alias.
    """
    if use_safetensors is not None:
        requested = "safetensors" if use_safetensors else "protobuf"
        if checkpoint_format != "auto" and checkpoint_format != requested:
            raise ValueError("Conflicting checkpoint format options")
        checkpoint_format = requested
    if checkpoint_format not in {"auto", "safetensors", "protobuf"}:
        raise ValueError(f"Unsupported checkpoint_format: {checkpoint_format!r}")

    resolve_kwargs = {
        key: kwargs[key]
        for key in (
            "revision",
            "cache_dir",
            "token",
            "local_files_only",
            "repo_type",
        )
        if key in kwargs
    }
    if checkpoint_format in {"auto", "safetensors"}:
        try:
            checkpoint = resolve_safetensors_checkpoint(model_id, **resolve_kwargs)
        except FileNotFoundError:
            if checkpoint_format == "safetensors":
                raise
        else:
            requested_path = pathlib.Path(model_id).expanduser()
            config_source = (
                checkpoint.parent
                if requested_path.exists() and requested_path.is_file()
                else model_id
            )
            config = leela_config.LeelaNetConfig.from_pretrained(
                config_source, **resolve_kwargs
            )
            state_dict = load_safetensors_file(checkpoint, device=device)
            try:
                model = build_with_state_dict(
                    lambda: _model_from_config(config), state_dict
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Checkpoint did not strictly match the Lc0 architecture: {exc}"
                ) from exc
            logging.info("Loaded safetensors checkpoint from %s", checkpoint)
            model.to(device=device)
            return _set_mode(model, is_eval)

    return _load_protobuf_checkpoint(
        model_id,
        is_eval=is_eval,
        resolve_kwargs=resolve_kwargs,
    )


def _load_protobuf_checkpoint(
    model_id: str | pathlib.Path,
    *,
    is_eval: bool,
    resolve_kwargs: dict[str, tp.Any],
) -> Lc0Model:
    """Invoke the bundled legacy adapter only for an explicit protobuf path."""
    from chessnets.models._pretrained import resolve_pretrained_file

    path = pathlib.Path(model_id).expanduser()
    if path.is_file():
        model_path = path.resolve()
        config_root = model_path.parent
    elif path.is_dir():
        model_path = path / "model.pb.gz"
        config_root = path
    else:
        model_path = resolve_pretrained_file(model_id, "model.pb.gz", **resolve_kwargs)
        config_root = model_path.parent
    if not model_path.is_file():
        raise FileNotFoundError(f"Could not find a protobuf checkpoint at {model_path}")

    config_path = config_root / "config.json"
    if not config_path.is_file():
        config_path = config_root / "config.yaml"
    if not config_path.is_file():
        config_path = resolve_pretrained_file(model_id, "config.yaml", **resolve_kwargs)
    return load_local_model(model_path, config_path, is_eval=is_eval)


def load_local_model(
    model_path: str | pathlib.Path,
    config_path: str | pathlib.Path,
    is_eval: bool = True,
) -> Lc0Model:
    """Load a legacy Lc0 protobuf checkpoint through the compatibility adapter."""
    return _get_protobuf_adapter().load_model(model_path, config_path, is_eval=is_eval)


def _get_protobuf_adapter():
    """Find the separately installed legacy protobuf adapter."""
    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):
        candidates = entry_points.select(
            group="chessnets.lc0_checkpoint_loaders", name="protobuf"
        )
    else:  # pragma: no cover - compatibility with older importlib metadata
        candidates = [
            point
            for point in entry_points.get("chessnets.lc0_checkpoint_loaders", ())
            if point.name == "protobuf"
        ]
    candidates = list(candidates)
    if len(candidates) > 1:
        raise RuntimeError("Multiple Lc0 protobuf adapters are installed")
    if candidates:
        return candidates[0].load()
    try:
        module = importlib.import_module("chessnets_lc0_protobuf.adapter")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading .pb.gz checkpoints requires the separately licensed "
            "(GPL-3.0-or-later) 'chessnets-lc0-protobuf' adapter; install it "
            'with `pip install "chessnets[lc0-protobuf]"`. Safetensors '
            "checkpoints do not require it."
        ) from exc
    return module.adapter


__all__ = ["load_local_model", "load_pretrained_model"]
