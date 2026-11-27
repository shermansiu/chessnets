"""Convert Lc0/Maia protobuf checkpoints to canonical safetensors repos."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import typing as tp

import torch
from safetensors.torch import save_file

from chessnets.models._pretrained import resolve_pretrained_file
from chessnets.models.lc0.configuration_lc0 import (
    Lc0InferenceConfig,
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(
    source: str | pathlib.Path, resolve_kwargs: dict[str, tp.Any]
) -> pathlib.Path:
    path = pathlib.Path(source).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidate = path / "model.pb.gz"
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Could not find 'model.pb.gz' in {path}")
    return resolve_pretrained_file(source, "model.pb.gz", **resolve_kwargs)


def _resolve_config(
    source: str | pathlib.Path,
    source_path: pathlib.Path,
    config_path: str | pathlib.Path | None,
    resolve_kwargs: dict[str, tp.Any],
) -> tuple[pathlib.Path, Lc0InferenceConfig]:
    if config_path is not None:
        path = pathlib.Path(config_path).expanduser().resolve()
    else:
        path = source_path.parent / "config.json"
        if not path.is_file():
            path = source_path.parent / "config.yaml"
        if not path.is_file():
            for filename in ("config.json", "config.yaml"):
                try:
                    path = resolve_pretrained_file(source, filename, **resolve_kwargs)
                    break
                except FileNotFoundError:
                    continue
            else:
                raise FileNotFoundError(
                    "No config was found; pass --config for a protobuf-only directory"
                )

    if path.suffix == ".json":
        inference_config = Lc0InferenceConfig.from_json(path)
    else:
        inference_config = Lc0InferenceConfig.from_yaml(path)
    return path, inference_config


def convert_checkpoint(
    source: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    config_path: str | pathlib.Path | None = None,
    retain_protobuf: bool = True,
    dry_run: bool = False,
    revision: str | None = None,
    cache_dir: str | pathlib.Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> dict[str, tp.Any]:
    """Decode a LINEAR16 protobuf into a float32 safetensors checkpoint."""
    resolve_kwargs = {
        "revision": revision,
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
    }
    source_path = _resolve_source(source, resolve_kwargs)
    resolved_config_path, config = _resolve_config(
        source, source_path, config_path, resolve_kwargs
    )
    output = pathlib.Path(output_dir).expanduser().resolve()
    from chessnets.loaders.load_lc0_model import _get_protobuf_adapter

    source_format = _get_protobuf_adapter().inspect_checkpoint(source_path)
    report: dict[str, tp.Any] = {
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_config": str(resolved_config_path),
        "source_config_sha256": _sha256(resolved_config_path),
        "output": str(output),
        "architecture": config.architectures[0],
        **source_format,
    }
    if dry_run:
        return report

    from chessnets.loaders.load_lc0_model import load_local_model

    model = load_local_model(source_path, resolved_config_path, is_eval=True).float()
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    non_float = {
        name: str(tensor.dtype)
        for name, tensor in state_dict.items()
        if not tensor.is_floating_point()
    }
    float_dtypes = {
        str(tensor.dtype)
        for tensor in state_dict.values()
        if tensor.is_floating_point()
    }
    if float_dtypes != {str(torch.float32)}:
        raise RuntimeError(f"Conversion produced non-float32 tensors: {float_dtypes}")

    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "model.safetensors"
    save_file(
        state_dict,
        str(weights_path),
        metadata={
            "format": "pt",
            "source_container_format": "lc0-protobuf",
            "source_weight_encoding": "LINEAR16",
            "decoded_tensor_dtype": "float32",
        },
    )
    config.save_pretrained(output)
    if retain_protobuf:
        original_dir = output / "original"
        original_dir.mkdir(exist_ok=True)
        retained_path = original_dir / "model.pb.gz"
        if source_path.resolve() != retained_path.resolve():
            shutil.copy2(source_path, retained_path)
        report["retained_source"] = str(retained_path)
    if resolved_config_path.suffix in {".yaml", ".yml"}:
        training_dir = output / "training"
        training_dir.mkdir(exist_ok=True)
        retained_config = training_dir / "original_lczero_training_config.yaml"
        if resolved_config_path.resolve() != retained_config.resolve():
            shutil.copy2(resolved_config_path, retained_config)
        report["retained_source_config"] = str(retained_config)
    report.update(
        {
            "safetensors": str(weights_path),
            "safetensors_sha256": _sha256(weights_path),
            "tensor_count": len(state_dict),
            "non_float_tensors": non_float,
        }
    )
    provenance = {
        "schema_version": 1,
        "source_reference": str(source),
        "source_sha256": report["source_sha256"],
        "source_config_sha256": report["source_config_sha256"],
        "source_container_format": report["source_container_format"],
        "source_weight_encoding": report["source_weight_encoding"],
        "decoded_tensor_dtype": report["decoded_tensor_dtype"],
        "architecture": report["architecture"],
        "safetensors_sha256": report["safetensors_sha256"],
        "tensor_count": report["tensor_count"],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        epilog=(
            "Requires the separately distributed chessnets-lc0-protobuf adapter; "
            "see plugins/lc0-protobuf in the ChessNets source repository."
        ),
    )
    parser.add_argument("source", help="Model repo, directory, or model.pb.gz path")
    parser.add_argument("output", help="Output model repository directory")
    parser.add_argument("--config", help="Config JSON/YAML for protobuf-only sources")
    parser.add_argument(
        "--no-retain-protobuf",
        action="store_true",
        help="Do not copy the source to original/model.pb.gz",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--token")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    args = _parser(prog).parse_args(argv)
    from chessnets.loaders.load_lc0_model import _get_protobuf_adapter

    try:
        _get_protobuf_adapter()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"error: {exc}") from None
    report = convert_checkpoint(
        args.source,
        args.output,
        config_path=args.config,
        retain_protobuf=not args.no_retain_protobuf,
        dry_run=args.dry_run,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=args.token,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["convert_checkpoint", "main"]
