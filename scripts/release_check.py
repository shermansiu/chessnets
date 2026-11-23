#!/usr/bin/env python3
"""Check that the repository is ready to release a tag, and extract its notes.

Run by .github/workflows/release.yml before anything is built or uploaded.
It fails unless:

- both packages share one version (scripts/sync_versions.py);
- the tag is exactly ``v<version>``;
- neither package has a ``Private ::`` classifier;
- CITATION.cff and the README's BibTeX entry give the same version;
- CHANGELOG.md has a non-empty ``## [<version>] - YYYY-MM-DD`` section.

Usage:
  scripts/release_check.py --tag v0.2.0 [--notes-out notes.md]
  scripts/release_check.py --draft [--notes-out notes.md]

--draft is for the release workflow's manual dry run between releases: the
tag is taken from the package version, and an undated changelog is allowed
(the notes come from the [Unreleased] section instead). Every other check
still applies.

With --notes-out, that changelog section is written to the file for the
GitHub Release. If GITHUB_OUTPUT is set, ``version`` and ``prerelease``
(true for a, b, rc and dev versions) are appended to it.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "sync_versions", ROOT / "scripts" / "sync_versions.py"
)
assert _spec is not None and _spec.loader is not None
sync_versions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_versions)

_PRERELEASE = re.compile(r"(a|b|rc|\.dev)\d+")


def changelog_section(text: str, version: str) -> str:
    """Return the body of the ``## [version] - date`` section, or raise."""
    heading = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}[ \t]*$", re.MULTILINE
    )
    match = heading.search(text)
    if match is None:
        raise ValueError(
            f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' heading; rename "
            "the [Unreleased] section for this release"
        )
    rest = text[match.end() :]
    # The section ends at the next release heading or the link references.
    end = re.search(r"^(## |\[[^\]]+\]: )", rest, re.MULTILINE)
    body = (rest[: end.start()] if end else rest).strip()
    if not body:
        raise ValueError(f"the CHANGELOG.md section for {version} is empty")
    return body + "\n"


def unreleased_section(text: str) -> str:
    """Return the body of ``## [Unreleased]``, or a placeholder if it is empty."""
    match = re.search(r"^## \[Unreleased\][ \t]*$", text, re.MULTILINE)
    body = ""
    if match:
        rest = text[match.end() :]
        end = re.search(r"^(## |\[[^\]]+\]: )", rest, re.MULTILINE)
        body = (rest[: end.start()] if end else rest).strip()
    return (body or "(No unreleased changes recorded.)") + "\n"


def check(
    tag: str | None, root: pathlib.Path = ROOT, *, draft: bool = False
) -> tuple[str, str]:
    """Return ``(version, notes)``; raise ValueError listing every problem."""
    errors = list(sync_versions.check(root))
    version = sync_versions.core_version(root)
    if tag is None:
        tag = f"v{version}"
    if tag != f"v{version}":
        errors.append(f"tag {tag!r} does not match the package version v{version}")
    for path in (root / "pyproject.toml", root / "plugins/lc0-protobuf/pyproject.toml"):
        project = tomllib.loads(path.read_text())["project"]
        if any(c.startswith("Private ::") for c in project.get("classifiers", [])):
            errors.append(f"{project['name']} has a 'Private ::' classifier")
    for name, pattern in (
        ("CITATION.cff", r"^version: *\"?([^\s\"]+)\"?$"),
        ("README.md", r"^ *version *= *\{([^}]*)\},?$"),
    ):
        path = root / name
        found = (
            re.findall(pattern, path.read_text(), re.MULTILINE) if path.exists() else []
        )
        if found != [version]:
            errors.append(f"{name} gives version {found}, expected [{version!r}]")
    notes = ""
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        errors.append("CHANGELOG.md is missing")
    else:
        text = changelog.read_text()
        try:
            notes = changelog_section(text, version)
        except ValueError as error:
            if draft:
                notes = unreleased_section(text)
            else:
                errors.append(str(error))
    if errors:
        raise ValueError("\n".join(errors))
    return version, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--tag", help="the release tag, e.g. v0.2.0")
    target.add_argument(
        "--draft", action="store_true", help="dry run: allow an undated changelog"
    )
    parser.add_argument("--notes-out", type=pathlib.Path)
    args = parser.parse_args(argv)

    try:
        version, notes = check(args.tag, draft=args.draft)
    except ValueError as error:
        print("Not ready to release:", file=sys.stderr)
        for line in str(error).splitlines():
            print(f"  {line}", file=sys.stderr)
        return 1

    if args.notes_out:
        args.notes_out.write_text(notes)
    prerelease = "true" if _PRERELEASE.search(version) else "false"
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as file:
            file.write(f"version={version}\nprerelease={prerelease}\n")
    what = f"Dry run for v{version}" if args.draft else f"Ready to release {args.tag}"
    print(f"{what} (prerelease: {prerelease}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
