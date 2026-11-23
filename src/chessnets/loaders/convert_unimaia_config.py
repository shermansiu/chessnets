"""Consolidate a legacy UniMaia YAML pair into one canonical config.json."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shutil
import typing as tp

from chessnets.models.lc0_senet.configuration_lc0_senet import LeelaInferenceConfig
from chessnets.models.unimaia.configuration_unimaia import UniMaiaConfig


def migrate_unimaia_config(
    repository: str | pathlib.Path,
    *,
    write: bool = False,
    archive_yaml: bool = False,
) -> dict[str, tp.Any]:
    """Build canonical config data and optionally update a local model repo."""
    root = pathlib.Path(repository).expanduser().resolve()
    legacy_path = root / "config.yaml"
    if not legacy_path.is_file():
        raise FileNotFoundError(f"Could not find {legacy_path}")
    config = UniMaiaConfig.from_yaml(legacy_path)
    if not config.lc0_config_path:
        raise ValueError("Legacy UniMaia config does not name lc0_config_path")
    lc0_path = pathlib.Path(config.lc0_config_path)
    if not lc0_path.is_absolute():
        lc0_path = root / lc0_path
    lc0_config = LeelaInferenceConfig.from_yaml(lc0_path)
    config = dataclasses.replace(
        config,
        lc0_config=lc0_config.to_dict(),
        lc0_config_path=None,
    )
    canonical = config.to_dict()
    report: dict[str, tp.Any] = {
        "repository": str(root),
        "config_json": str(root / "config.json"),
        "legacy_config": str(legacy_path),
        "legacy_lc0_config": str(lc0_path),
        "architecture": canonical["architectures"][0],
        "lc0_architecture": lc0_config.architectures[0],
        "write": write,
        "archive_yaml": archive_yaml,
    }
    if not write:
        report["canonical_config"] = canonical
        return report

    (root / "config.json").write_text(
        json.dumps(canonical, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if archive_yaml:
        legacy_target = root / "legacy" / "inference_config.yaml"
        lc0_target = root / "training" / "original_lczero_training_config.yaml"
        for target in (legacy_target, lc0_target):
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}")
            target.parent.mkdir(exist_ok=True)
        shutil.move(legacy_path, legacy_target)
        shutil.move(lc0_path, lc0_target)
        report["archived_legacy_config"] = str(legacy_target)
        report["archived_lc0_config"] = str(lc0_target)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--archive-yaml", action="store_true")
    args = parser.parse_args(argv)
    if args.archive_yaml and not args.write:
        parser.error("--archive-yaml requires --write")
    report = migrate_unimaia_config(
        args.repository, write=args.write, archive_yaml=args.archive_yaml
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["main", "migrate_unimaia_config"]
