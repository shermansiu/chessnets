#!/usr/bin/env python3
"""Keep the chessnets and chessnets-lc0-protobuf versions in lockstep.

The core ``[project].version`` in ``pyproject.toml`` is the single source of
truth. This script derives three other fields from it:

- the plugin's ``[project].version``;
- the plugin's ``chessnets==<version>`` dependency;
- the core ``lc0-protobuf`` extra's ``chessnets-lc0-protobuf==<version>``.

Usage:
  scripts/sync_versions.py                  check only; exit 1 if out of sync
  scripts/sync_versions.py --write          copy the core version everywhere
  scripts/sync_versions.py --set 0.2.0      set a new version everywhere
  scripts/sync_versions.py --bump minor     bump major/minor/patch everywhere
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "pyproject.toml"
PLUGIN = ROOT / "plugins" / "lc0-protobuf" / "pyproject.toml"
CORE_NAME = "chessnets"
PLUGIN_NAME = "chessnets-lc0-protobuf"
PLUGIN_EXTRA = "lc0-protobuf"

_VERSION = re.compile(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?")
_VERSION_LINE = re.compile(r'^version = "[^"]*"$', re.MULTILINE)
# A quoted requirement on exactly this distribution name, with any specifier.
_CORE_REQUIREMENT = re.compile(r'"chessnets\s*(?:[<>=!~][^"]*)?"')
_PLUGIN_REQUIREMENT = re.compile(r'"chessnets-lc0-protobuf\s*(?:[<>=!~][^"]*)?"')


def core_version(root: pathlib.Path = ROOT) -> str:
    project = tomllib.loads((root / CORE.relative_to(ROOT)).read_text())["project"]
    return project["version"]


def check(root: pathlib.Path = ROOT) -> list[str]:
    """Return a description of every field that disagrees with the core version."""
    core = tomllib.loads((root / CORE.relative_to(ROOT)).read_text())["project"]
    plugin = tomllib.loads((root / PLUGIN.relative_to(ROOT)).read_text())["project"]
    version = core["version"]
    errors = []
    if plugin["version"] != version:
        errors.append(
            f"{PLUGIN_NAME} version is {plugin['version']}, expected {version}"
        )
    expected = f"{CORE_NAME}=={version}"
    core_deps = [
        dep
        for dep in plugin.get("dependencies", [])
        if re.match(rf"{CORE_NAME}\s*(?:[<>=!~]|$)", dep)
    ]
    if core_deps != [expected]:
        errors.append(f"{PLUGIN_NAME} depends on {core_deps}, expected [{expected!r}]")
    expected = f"{PLUGIN_NAME}=={version}"
    extra = core.get("optional-dependencies", {}).get(PLUGIN_EXTRA)
    if extra != [expected]:
        errors.append(
            f"{CORE_NAME}[{PLUGIN_EXTRA}] is {extra}, expected [{expected!r}]"
        )
    return errors


def _replace_one(text: str, pattern: re.Pattern[str], new: str, where: str) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise SystemExit(
            f"error: expected one match for {pattern.pattern!r} in {where}, "
            f"found {len(matches)}"
        )
    return pattern.sub(new, text)


def write(version: str, root: pathlib.Path = ROOT) -> None:
    """Set ``version`` in all four fields, preserving the rest of each file."""
    if not _VERSION.fullmatch(version):
        raise SystemExit(f"error: not a supported version string: {version!r}")
    core_path = root / CORE.relative_to(ROOT)
    plugin_path = root / PLUGIN.relative_to(ROOT)

    text = core_path.read_text()
    text = _replace_one(text, _VERSION_LINE, f'version = "{version}"', core_path.name)
    text = _replace_one(
        text, _PLUGIN_REQUIREMENT, f'"{PLUGIN_NAME}=={version}"', core_path.name
    )
    core_path.write_text(text)

    text = plugin_path.read_text()
    text = _replace_one(text, _VERSION_LINE, f'version = "{version}"', str(plugin_path))
    text = _replace_one(
        text, _CORE_REQUIREMENT, f'"{CORE_NAME}=={version}"', str(plugin_path)
    )
    plugin_path.write_text(text)

    errors = check(root)
    if errors:
        raise SystemExit("error: files still out of sync:\n  " + "\n  ".join(errors))


def bump(version: str, part: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise SystemExit(
            f"error: --bump needs a plain X.Y.Z version, not {version!r}; use --set"
        )
    major, minor, patch = map(int, match.groups())
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--write", action="store_true", help="propagate the core version"
    )
    action.add_argument("--set", metavar="VERSION", help="set a new version")
    action.add_argument("--bump", choices=("major", "minor", "patch"))
    args = parser.parse_args(argv)

    current = core_version()
    if args.write or args.set or args.bump:
        new = args.set or (bump(current, args.bump) if args.bump else current)
        write(new)
        print(f"{CORE_NAME} and {PLUGIN_NAME} are now at {new}")
        return 0

    errors = check()
    if errors:
        print("Versions are out of sync (fix with scripts/sync_versions.py --write):")
        for error in errors:
            print(f"  {error}")
        return 1
    print(f"{CORE_NAME} and {PLUGIN_NAME} are both at {current}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
