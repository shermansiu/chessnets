#!/usr/bin/env python3
"""Check that built wheels and sdists contain only code their license allows.

Each artifact is checked against the ``License-Expression`` in its own
metadata, so the same check works for the Apache-2.0/MIT core package and the
GPL-3.0-or-later lc0-protobuf plugin:

- every ``SPDX-License-Identifier`` header must name only licenses in that
  expression;
- an artifact whose expression has no GPL-family license must not contain GPL
  license text, import protobuf or generated ``*_pb2`` code, or include the
  plugin package.

Passing is evidence about file contents, not a legal conclusion.
"""

from __future__ import annotations

import argparse
import email.parser
import pathlib
import re
import tarfile
import typing as tp
import zipfile

SPDX_HEADER = re.compile(r"SPDX-License-Identifier:\s*([^\n*\"'#]+)")
EXPRESSION_OPERATORS = {"AND", "OR", "WITH"}
COPYLEFT_PREFIXES = ("GPL-", "LGPL-", "AGPL-")
COPYLEFT_TEXT = re.compile(
    r"GNU\s+(?:Lesser\s+|Library\s+|Affero\s+)?General\s+Public\s+License",
    re.IGNORECASE,
)
PROTOBUF_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:google\.protobuf\b|\S*_pb2\b|\S+\s+import\s+\S*_pb2\b)",
    re.MULTILINE,
)
# Packages that must never ship inside a non-copyleft artifact.
COPYLEFT_PACKAGES = ("chessnets_lc0_protobuf/",)


def license_ids(expression: str) -> set[str]:
    tokens = re.split(r"[\s()]+", expression)
    return {token for token in tokens if token and token not in EXPRESSION_OPERATORS}


def is_copyleft(license_id: str) -> bool:
    return license_id.startswith(COPYLEFT_PREFIXES)


def _members(path: pathlib.Path) -> tp.Iterator[tuple[str, bytes]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    yield info.filename, archive.read(info)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    file = archive.extractfile(member)
                    assert file is not None
                    yield member.name, file.read()
        return
    raise ValueError(f"Unsupported distribution artifact: {path}")


def _text(data: bytes) -> str | None:
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _strip_sdist_root(name: str) -> str:
    """Make sdist member paths comparable to wheel paths."""
    parts = name.split("/", 1)
    return parts[1] if len(parts) == 2 else name


def check_artifact(path: str | pathlib.Path) -> list[str]:
    """Return one message per problem found in a wheel or sdist."""
    path = pathlib.Path(path)
    files = dict(_members(path))
    metadata_name = next(
        (
            name
            for name in files
            if name.endswith(".dist-info/METADATA")
            or (name.count("/") == 1 and name.endswith("/PKG-INFO"))
        ),
        None,
    )
    if metadata_name is None:
        return ["no METADATA or PKG-INFO found"]
    metadata = email.parser.Parser().parsestr(files[metadata_name].decode())
    expression = metadata.get("License-Expression")
    if not expression:
        return [f"{metadata_name} has no License-Expression"]
    allowed = license_ids(expression)
    copyleft_allowed = any(is_copyleft(license_id) for license_id in allowed)

    is_sdist = metadata_name.endswith("/PKG-INFO")
    problems = []
    for name, data in sorted(files.items()):
        relative = _strip_sdist_root(name) if is_sdist else name
        if not copyleft_allowed and relative.startswith(
            COPYLEFT_PACKAGES + tuple(f"src/{pkg}" for pkg in COPYLEFT_PACKAGES)
        ):
            problems.append(
                f"{name}: copyleft plugin package in a {expression} artifact"
            )
            continue
        text = _text(data)
        if text is None:
            continue
        for header in SPDX_HEADER.findall(text):
            outside = license_ids(header.strip()) - allowed
            if outside:
                problems.append(
                    f"{name}: SPDX {header.strip()!r} is not covered by "
                    f"License-Expression {expression!r}"
                )
        if copyleft_allowed:
            continue
        if COPYLEFT_TEXT.search(text):
            problems.append(f"{name}: contains GNU license text")
        if name.endswith(".py") and PROTOBUF_IMPORT.search(text):
            problems.append(f"{name}: imports protobuf or generated *_pb2 code")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("artifacts", nargs="+", type=pathlib.Path)
    args = parser.parse_args(argv)
    failed = False
    for artifact in args.artifacts:
        problems = check_artifact(artifact)
        if problems:
            failed = True
            print(f"{artifact}: FAILED")
            for problem in problems:
                print(f"  {problem}")
        else:
            print(f"{artifact}: licenses OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
