"""Helpers shared by every model card family."""

from __future__ import annotations

import difflib
import pathlib
import subprocess

_HF_PREFIXES = ("git@hf.co:", "https://huggingface.co/", "http://huggingface.co/")


def repository_id(root: pathlib.Path) -> str:
    """Return the Hugging Face ``owner/name`` from the Git remote, else the dir name."""
    try:
        remote = subprocess.check_output(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return root.name
    for prefix in _HF_PREFIXES:
        if remote.startswith(prefix):
            return remote.removeprefix(prefix).removesuffix(".git")
    return root.name


def read_readme(root: pathlib.Path, *, required: bool) -> str:
    """Read README.md with newlines normalized to LF."""
    readme = root / "README.md"
    if not readme.is_file():
        if required:
            raise FileNotFoundError(f"Could not find {readme}")
        return ""
    return readme.read_text(encoding="utf-8").replace("\r\n", "\n")


def replace_tail(text: str, heading: str, generated: str) -> str:
    """Replace everything from ``heading`` onward (or append) with ``generated``."""
    count = text.count(heading)
    if count > 1:
        raise ValueError(f"Model card contains duplicate {heading!r} headings")
    head = text[: text.index(heading)] if count else text
    return head.rstrip() + "\n\n" + generated + "\n"


def write_or_preview(
    root: pathlib.Path, current: str, updated: str, *, write: bool
) -> bool:
    """Write ``updated`` or print a diff against ``current``; return whether it changed."""
    readme = root / "README.md"
    changed = updated != current
    if write:
        # Always write LF so Git does not flag legacy CRLF cards as whitespace errors.
        readme.write_text(updated, encoding="utf-8", newline="\n")
    elif changed:
        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(readme),
            tofile=f"{readme} (proposed)",
        )
        print("".join(diff), end="")
    else:
        print(f"{readme}: model card is already current")
    return changed
