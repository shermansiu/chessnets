"""Convert an upstream Maia 2 training checkpoint to exact safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import typing as tp

import torch
from safetensors.torch import load_file, save_file

from chessnets.models.maia2.configuration_maia2 import Config


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inference_state(checkpoint: tp.Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Maia 2 checkpoint must contain a mapping")
    raw = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Maia 2 checkpoint has no model_state_dict")
    state: dict[str, torch.Tensor] = {}
    for original_name, tensor in raw.items():
        if not isinstance(original_name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("Maia 2 state dict must contain named tensors only")
        name = original_name.removeprefix("module.")
        if name in state:
            raise ValueError(f"Duplicate normalized state-dict key: {name}")
        state[name] = tensor.detach().cpu().contiguous()
    return state


def _assert_identical(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> None:
    if expected.keys() != actual.keys():
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        raise RuntimeError(
            f"Safetensors keys differ; missing={missing}, unexpected={unexpected}"
        )
    for name, tensor in expected.items():
        restored = actual[name]
        if tensor.dtype != restored.dtype or tensor.shape != restored.shape:
            raise RuntimeError(f"Safetensors metadata differs for {name!r}")
        if not torch.equal(tensor, restored):
            raise RuntimeError(f"Safetensors tensor differs for {name!r}")


def _assert_outputs_identical(
    config: Config,
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    from chessnets.models.maia2.modeling_maia2 import Maia2Model

    original = Maia2Model.from_config(config)
    converted = Maia2Model.from_config(config)
    original.load_state_dict(expected, strict=True)
    converted.load_state_dict(actual, strict=True)
    original.eval()
    converted.eval()
    boards = torch.zeros((2, config.input_channels, 8, 8), dtype=torch.float32)
    elos_self = torch.tensor([0, 1], dtype=torch.int32)
    elos_oppo = torch.tensor([1, 0], dtype=torch.int32)
    with torch.inference_mode():
        expected_outputs = original(boards, elos_self, elos_oppo)
        actual_outputs = converted(boards, elos_self, elos_oppo)
    for index, (left, right) in enumerate(zip(expected_outputs, actual_outputs)):
        if not torch.equal(left, right):
            raise RuntimeError(f"Converted model output {index} is not bit-identical")


def convert_maia2_checkpoint(
    checkpoint: str | pathlib.Path,
    training_config: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    time_control: tp.Literal["blitz", "rapid"],
    source_url: str | None = None,
    training_config_url: str | None = None,
    write: bool = True,
) -> dict[str, tp.Any]:
    """Convert and verify one Maia 2 checkpoint."""
    checkpoint_path = pathlib.Path(checkpoint).expanduser().resolve()
    config_path = pathlib.Path(training_config).expanduser().resolve()
    output = pathlib.Path(output_dir).expanduser().resolve()
    if time_control not in {"blitz", "rapid"}:
        raise ValueError("time_control must be 'blitz' or 'rapid'")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    legacy_config = Config.from_yaml(config_path)
    values = legacy_config.to_dict()
    values["time_control"] = time_control
    config = Config.from_dict(values)
    checkpoint_hash = _sha256(checkpoint_path)
    training_config_hash = _sha256(config_path)
    report: dict[str, tp.Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "config": config.to_dict(),
        "output_dir": str(output),
        "time_control": time_control,
        "training_config_sha256": training_config_hash,
        "write": write,
    }
    if source_url:
        report["source_url"] = source_url
    if training_config_url:
        report["training_config_url"] = training_config_url
    if not write:
        return report

    output.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "model.safetensors", "provenance.json"):
        if (output / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {output / filename}")

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = _inference_state(checkpoint_data)
    model_path = output / "model.safetensors"
    save_file(
        state,
        model_path,
        metadata={
            "format": "pt",
            "source_checkpoint_sha256": checkpoint_hash,
            "time_control": time_control,
        },
    )
    restored = load_file(model_path, device="cpu")
    _assert_identical(state, restored)
    _assert_outputs_identical(config, state, restored)
    config.save_pretrained(output)

    provenance = {
        "conversion": {
            "key_normalization": (
                "renamed each state-dict key from module.<name> to <name> by "
                "removing the PyTorch DataParallel prefix"
            ),
            "output_format": "safetensors",
            "tensor_equality": "torch.equal for every tensor",
            "output_equality": "torch.equal for policy, auxiliary, and value outputs",
        },
        "source": {
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_hash,
            "training_config_filename": config_path.name,
            "training_config_sha256": training_config_hash,
        },
        "tensor_count": len(state),
        "time_control": time_control,
    }
    if source_url:
        provenance["source"]["url"] = source_url
    if training_config_url:
        provenance["source"]["training_config_url"] = training_config_url
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report.update(
        model_safetensors_sha256=_sha256(model_path),
        tensor_count=len(state),
        tensors_identical=True,
        outputs_identical=True,
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("training_config")
    parser.add_argument("output_dir")
    parser.add_argument("--time-control", choices=("blitz", "rapid"), required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--training-config-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = convert_maia2_checkpoint(
        args.checkpoint,
        args.training_config,
        args.output_dir,
        time_control=args.time_control,
        source_url=args.source_url,
        training_config_url=args.training_config_url,
        write=not args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["convert_maia2_checkpoint", "main"]
