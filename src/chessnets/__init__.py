"""ChessNets: chess-playing and human move-imitation models.

``AutoPipeline`` is imported lazily so that ``import chessnets`` (and the
``chessnets`` command line) stay fast; PyTorch loads on first use.
"""

from __future__ import annotations

import typing as _tp
from importlib import metadata as _metadata

if _tp.TYPE_CHECKING:
    from .pipelines.auto_pipeline import AutoPipeline

try:
    __version__ = _metadata.version("chessnets")
except _metadata.PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0+unknown"


def __getattr__(name: str) -> _tp.Any:
    if name == "AutoPipeline":
        from .pipelines.auto_pipeline import AutoPipeline

        return AutoPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AutoPipeline", "__version__"]
