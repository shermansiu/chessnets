"""Small helpers shared by model ``from_pretrained`` implementations.

The project deliberately does not depend on ``transformers`` for its chess
models, but it still benefits from the same local-path/Hugging Face resolution
semantics.  Keeping that resolution here prevents each model family from
growing a subtly different loader API.
"""

from __future__ import annotations

import pathlib
import typing as tp

from huggingface_hub.utils import EntryNotFoundError


def resolve_pretrained_file(
    model_id_or_path: str | pathlib.Path,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | pathlib.Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    repo_type: str | None = None,
) -> pathlib.Path:
    """Resolve an artifact in a local checkpoint directory or HF repository."""
    root = pathlib.Path(model_id_or_path).expanduser()
    if root.is_dir():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Could not find {filename!r} in {root}")
        return path

    from huggingface_hub import hf_hub_download

    kwargs: dict[str, tp.Any] = {
        "repo_id": str(model_id_or_path),
        "filename": filename,
        "revision": revision,
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
    }
    if repo_type is not None:
        kwargs["repo_type"] = repo_type
    try:
        return pathlib.Path(hf_hub_download(**kwargs))
    except EntryNotFoundError as exc:
        # Normalize a missing Hub artifact to the same exception used for a
        # missing file in a local checkpoint directory. Other Hub errors,
        # including authentication and connectivity failures, still propagate.
        raise FileNotFoundError(
            f"Could not find {filename!r} in Hugging Face repository "
            f"{model_id_or_path!s}"
        ) from exc


__all__ = ["resolve_pretrained_file"]
