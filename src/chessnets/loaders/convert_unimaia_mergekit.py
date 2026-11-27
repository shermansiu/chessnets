"""Normalize legacy MergeKit UniMaia checkpoints for ChessNets.

MergeKit cast the non-persistent attention policy maps to bfloat16 and wrote
them into the checkpoint.  ChessNets constructs those maps canonically as
int64 buffers, so a compatible package keeps the original shards but removes
the six legacy buffers from the safetensors index.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import shutil
import tempfile
import typing as tp

import yaml
from safetensors import safe_open

from chessnets.models.lc0.configuration_lc0 import Lc0InferenceConfig
from chessnets.models.unimaia.configuration_unimaia import UniMaiaConfig

POLICY_MAP_KEYS = frozenset(
    f"{head}.attention_policy_map.idx"
    for head in (
        "policy_next",
        "policy_opponent",
        "policy_optimistic_st",
        "policy_soft",
        "policy_vanilla",
        "policy_vanilla_clone",
    )
)
LEGACY_DEFAULT_LLM_MODEL_ID = "Waterhorse/chessgpt-base-v1"
INDEX_FILENAME = "model.safetensors.index.json"
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _sha256(path: pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_shard_name(name: str) -> str:
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.name != name:
        raise ValueError(f"Unsafe safetensors shard name: {name!r}")
    if not name.endswith(".safetensors"):
        raise ValueError(f"Invalid safetensors shard name: {name!r}")
    return name


def _tensor_nbytes(shape: tp.Iterable[int], dtype: str) -> int:
    try:
        item_size = _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported safetensors dtype: {dtype!r}") from exc
    return math.prod(shape) * item_size


def inspect_mergekit_checkpoint(
    source: str | pathlib.Path,
    *,
    expected_tensor_count: int | None = 1541,
) -> dict[str, tp.Any]:
    """Validate a MergeKit index and return normalization metadata."""

    root = pathlib.Path(source).expanduser().resolve()
    index_path = root / INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(f"Could not find {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in {index_path}")
    if expected_tensor_count is not None and len(weight_map) != expected_tensor_count:
        raise ValueError(
            f"Expected {expected_tensor_count} indexed tensors, found {len(weight_map)}"
        )

    expected_by_shard: dict[str, set[str]] = {}
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard, str):
            raise ValueError(f"Invalid weight_map entry in {index_path}")
        expected_by_shard.setdefault(_safe_shard_name(shard), set()).add(key)

    actual_keys: set[str] = set()
    tensor_metadata: dict[str, dict[str, tp.Any]] = {}
    shard_tensor_counts: dict[str, int] = {}
    for shard, expected_keys in sorted(expected_by_shard.items()):
        shard_path = root / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as stream:
            shard_keys = set(stream.keys())
            duplicate = actual_keys.intersection(shard_keys)
            if duplicate:
                raise ValueError(
                    f"Duplicate tensors across shards: {sorted(duplicate)}"
                )
            actual_keys.update(shard_keys)
            shard_tensor_counts[shard] = len(shard_keys)
            for key in shard_keys:
                tensor = stream.get_slice(key)
                shape = list(tensor.get_shape())
                dtype = tensor.get_dtype()
                tensor_metadata[key] = {
                    "shape": shape,
                    "dtype": dtype,
                    "nbytes": _tensor_nbytes(shape, dtype),
                }

        missing = expected_keys.difference(shard_keys)
        if missing:
            raise ValueError(f"Shard {shard!r} is missing tensors: {sorted(missing)}")

    indexed_keys = set(weight_map)
    if actual_keys != indexed_keys:
        missing = sorted(indexed_keys.difference(actual_keys))
        extra = sorted(actual_keys.difference(indexed_keys))
        raise ValueError(f"Index/shard mismatch; missing={missing}, extra={extra}")
    policy_keys = {
        key for key in indexed_keys if key.endswith(".attention_policy_map.idx")
    }
    if policy_keys != POLICY_MAP_KEYS:
        raise ValueError(
            "Unexpected attention policy-map keys; "
            f"expected={sorted(POLICY_MAP_KEYS)}, found={sorted(policy_keys)}"
        )
    for key in sorted(POLICY_MAP_KEYS):
        metadata = tensor_metadata[key]
        if metadata["shape"] != [1858] or metadata["dtype"] != "BF16":
            raise ValueError(
                f"Unexpected legacy policy map {key}: "
                f"shape={metadata['shape']}, dtype={metadata['dtype']}"
            )

    total_size = sum(item["nbytes"] for item in tensor_metadata.values())
    declared_size = index.get("metadata", {}).get("total_size")
    if declared_size is not None and declared_size != total_size:
        raise ValueError(
            f"Index total_size is {declared_size}, calculated {total_size}"
        )
    excluded_size = sum(tensor_metadata[key]["nbytes"] for key in POLICY_MAP_KEYS)
    return {
        "source": str(root),
        "index": str(index_path),
        "tensor_count": len(indexed_keys),
        "normalized_tensor_count": len(indexed_keys - POLICY_MAP_KEYS),
        "total_size": total_size,
        "excluded_size": excluded_size,
        "normalized_total_size": total_size - excluded_size,
        "mergekit_version": index.get("metadata", {}).get("mergekit_version"),
        "shards": [
            {
                "name": shard,
                "tensor_count": shard_tensor_counts[shard],
                "size": (root / shard).stat().st_size,
            }
            for shard in sorted(expected_by_shard)
        ],
        "excluded_tensors": {
            key: tensor_metadata[key] for key in sorted(POLICY_MAP_KEYS)
        },
    }


def _link_or_copy(source: pathlib.Path, destination: pathlib.Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        raise ValueError(f"Unsupported link mode: {mode!r}")


def normalize_mergekit_checkpoint(
    source: str | pathlib.Path,
    training_config: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    lc0_config: str | pathlib.Path,
    text_model_directory: str | pathlib.Path,
    link_mode: str = "symlink",
    expected_tensor_count: int | None = 1541,
) -> dict[str, tp.Any]:
    """Create a ChessNets package while leaving MergeKit shards unchanged."""

    root = pathlib.Path(source).expanduser().resolve()
    config_path = pathlib.Path(training_config).expanduser().resolve()
    lc0_config_path = pathlib.Path(lc0_config).expanduser().resolve()
    text_model_root = pathlib.Path(text_model_directory).expanduser().resolve()
    destination = pathlib.Path(output).expanduser().resolve()
    inspection = inspect_mergekit_checkpoint(
        root, expected_tensor_count=expected_tensor_count
    )
    source_index = json.loads((root / INDEX_FILENAME).read_text(encoding="utf-8"))
    source_index_sha256 = _sha256(root / INDEX_FILENAME)
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find training config: {config_path}")
    if not lc0_config_path.is_file():
        raise FileNotFoundError(f"Could not find LC0 config: {lc0_config_path}")
    text_config_path = text_model_root / "config.json"
    if not text_config_path.is_file():
        raise FileNotFoundError(f"Could not find text model config: {text_config_path}")
    tokenizer_paths = [
        text_model_root / name
        for name in TOKENIZER_FILENAMES
        if (text_model_root / name).is_file()
    ]
    if not {path.name for path in tokenizer_paths}.intersection(
        {"tokenizer.json", "spiece.model", "sentencepiece.bpe.model"}
    ):
        raise FileNotFoundError(
            f"Could not find a serialized tokenizer in {text_model_root}"
        )
    training_config_sha256 = _sha256(config_path)
    lc0_config_sha256 = _sha256(lc0_config_path)
    text_asset_hashes = {
        "config.json": _sha256(text_config_path),
        **{path.name: _sha256(path) for path in tokenizer_paths},
    }

    if destination.exists():
        provenance_path = destination / "provenance.json"
        if provenance_path.is_file():
            existing = json.loads(provenance_path.read_text(encoding="utf-8"))
            existing_source = existing.get("source", {})
            if (
                existing_source.get("index_sha256") == source_index_sha256
                and existing_source.get("training_config_sha256")
                == training_config_sha256
                and existing_source.get("lc0_config_sha256") == lc0_config_sha256
                and existing_source.get("text_assets") == text_asset_hashes
            ):
                return {**existing, "reused": True}
        raise FileExistsError(f"Refusing to overwrite {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        shard_reports = []
        for shard in inspection["shards"]:
            source_shard = root / shard["name"]
            _link_or_copy(source_shard, temporary / shard["name"], link_mode)
            shard_reports.append(
                {
                    **shard,
                    "sha256": _sha256(source_shard),
                }
            )

        filtered_map = {
            key: shard
            for key, shard in source_index["weight_map"].items()
            if key not in POLICY_MAP_KEYS
        }
        normalized_index = {
            "metadata": {
                **source_index.get("metadata", {}),
                "total_size": inspection["normalized_total_size"],
                "chessnets_normalization": "drop-nonpersistent-policy-maps-v1",
            },
            "weight_map": dict(sorted(filtered_map.items())),
        }
        normalized_index_path = temporary / INDEX_FILENAME
        normalized_index_path.write_text(
            json.dumps(normalized_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        training_values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        training_values.setdefault("llm_model_id", LEGACY_DEFAULT_LLM_MODEL_ID)
        training_values["model_path"] = INDEX_FILENAME
        training_values["device"] = None
        training_values["tokenizer_path"] = "."
        training_values["llm_config_path"] = "text_model"
        config = UniMaiaConfig.from_dict(training_values)
        parsed_lc0_config = (
            Lc0InferenceConfig.from_json(lc0_config_path)
            if lc0_config_path.suffix.lower() == ".json"
            else Lc0InferenceConfig.from_yaml(lc0_config_path)
        )
        config = dataclasses.replace(
            config,
            lc0_config=parsed_lc0_config.to_dict(),
            lc0_config_path=None,
        )
        config.save_pretrained(temporary)
        text_config_destination = temporary / "text_model" / "config.json"
        text_config_destination.parent.mkdir()
        shutil.copy2(text_config_path, text_config_destination)
        for tokenizer_path in tokenizer_paths:
            shutil.copy2(tokenizer_path, temporary / tokenizer_path.name)
        legacy_dir = temporary / "legacy"
        legacy_dir.mkdir()
        shutil.copy2(config_path, legacy_dir / "training_config.yaml")
        shutil.copy2(
            lc0_config_path, legacy_dir / f"lc0_config{lc0_config_path.suffix}"
        )
        (legacy_dir / "resolved_inference_config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        provenance = {
            "schema_version": 2,
            "normalization": "drop-nonpersistent-policy-maps-v1",
            "source": {
                "directory_name": root.name,
                "index_sha256": source_index_sha256,
                "training_config_sha256": training_config_sha256,
                "lc0_config_sha256": lc0_config_sha256,
                "lc0_architecture": parsed_lc0_config.architectures[0],
                "text_assets": text_asset_hashes,
                "mergekit_version": inspection["mergekit_version"],
                "tensor_count": inspection["tensor_count"],
                "total_size": inspection["total_size"],
                "shards": shard_reports,
            },
            "output": {
                "index": INDEX_FILENAME,
                "index_sha256": _sha256(normalized_index_path),
                "tensor_count": inspection["normalized_tensor_count"],
                "total_size": inspection["normalized_total_size"],
                "config_sha256": _sha256(temporary / "config.json"),
                "link_mode": link_mode,
            },
            "excluded_tensors": inspection["excluded_tensors"],
            "reused": False,
        }
        (temporary / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return provenance
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("training_config", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--lc0-config",
        type=pathlib.Path,
        required=True,
        help="Published LC0 config.json/config.yaml to embed for offline loading",
    )
    parser.add_argument(
        "--text-model-directory",
        type=pathlib.Path,
        required=True,
        help="Directory containing the text config and tokenizer assets to embed",
    )
    parser.add_argument(
        "--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    parser.add_argument("--expected-tensor-count", type=int, default=1541)
    args = parser.parse_args(argv)
    report = normalize_mergekit_checkpoint(
        args.source,
        args.training_config,
        args.output,
        lc0_config=args.lc0_config,
        text_model_directory=args.text_model_directory,
        link_mode=args.link_mode,
        expected_tensor_count=args.expected_tensor_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "INDEX_FILENAME",
    "POLICY_MAP_KEYS",
    "inspect_mergekit_checkpoint",
    "main",
    "normalize_mergekit_checkpoint",
]
