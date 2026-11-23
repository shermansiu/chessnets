#!/usr/bin/env python3
"""Preview or write ChessNets model cards for first-party checkpoint repos.

Preview (a unified diff) is the default; pass --write to apply. The family is
detected per repository unless --family is given:

  maia2   config.json has model_type "maia2", or the name starts with maia2
  maia1   the directory name starts with "maia-"
  lc0     anything else

Lc0 and Maia-1 cards keep their hand-written top (frontmatter, source notes)
and regenerate everything from "## Use with ChessNets" onward. Maia 2 cards are
generated in full.

Examples:
  scripts/update_model_card.py ../checkpoints/maia-1100
  scripts/update_model_card.py --write ../checkpoints/*
  scripts/update_model_card.py --family lc0 --repository-id owner/net ./net
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from model_cards import common, lc0, maia2  # noqa: E402

FAMILIES = ("auto", "lc0", "maia1", "maia2")


def detect_family(root: pathlib.Path) -> str:
    config = root / "config.json"
    if config.is_file():
        if json.loads(config.read_text(encoding="utf-8")).get("model_type") == "maia2":
            return "maia2"
    name = root.name.lower()
    if name.startswith("maia2"):
        return "maia2"
    if name.startswith("maia-"):
        return "maia1"
    return "lc0"


def update_model_card(
    repository: str | pathlib.Path,
    *,
    write: bool = False,
    repository_id: str | None = None,
    family: str = "auto",
) -> bool:
    """Preview or write one repository's card; return whether it changed."""
    if family not in FAMILIES:
        raise ValueError(f"Unsupported model family: {family!r}")
    root = pathlib.Path(repository).expanduser().resolve()
    family = detect_family(root) if family == "auto" else family
    repo_id = repository_id or common.repository_id(root)
    if family == "maia2":
        return maia2.update(root, write=write, repository_id=repo_id)
    return lc0.update(root, write=write, repository_id=repo_id, maia1=family == "maia1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("repositories", nargs="+", type=pathlib.Path)
    parser.add_argument("--write", action="store_true", help="apply the changes")
    parser.add_argument("--family", choices=FAMILIES, default="auto")
    parser.add_argument(
        "--repository-id",
        help="Hugging Face owner/name (default: from the Git remote); one repo only",
    )
    args = parser.parse_args(argv)
    if args.repository_id and len(args.repositories) > 1:
        parser.error("--repository-id needs exactly one repository")
    for repository in args.repositories:
        update_model_card(
            repository,
            write=args.write,
            repository_id=args.repository_id,
            family=args.family,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
