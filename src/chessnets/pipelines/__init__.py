"""Prediction pipelines and registries.

The base pipeline types are lightweight and imported eagerly. The registry in
``auto_pipeline`` imports every built-in pipeline (and PyTorch), so its exports
load on first access.
"""

from __future__ import annotations

import typing as tp

from .base import (
    Pipeline,
    PipelineInputConfig,
    PipelineInputName,
    extract_domain_inputs,
)

if tp.TYPE_CHECKING:
    from .auto_pipeline import (
        ARCHITECTURE_INFOS,
        PIPELINE_INFOS,
        ArchitectureInfo,
        AutoPipeline,
        PipelineInfo,
        register_architecture,
    )
    from .pipeline_utils import DrawnPositionWarning

_AUTO_PIPELINE_EXPORTS = {
    "ARCHITECTURE_INFOS",
    "ArchitectureInfo",
    "AutoPipeline",
    "PIPELINE_INFOS",
    "PipelineInfo",
    "register_architecture",
}


def __getattr__(name: str) -> tp.Any:
    if name in _AUTO_PIPELINE_EXPORTS:
        from . import auto_pipeline

        return getattr(auto_pipeline, name)
    if name == "DrawnPositionWarning":
        from .pipeline_utils import DrawnPositionWarning

        return DrawnPositionWarning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AutoPipeline",
    "ARCHITECTURE_INFOS",
    "ArchitectureInfo",
    "DrawnPositionWarning",
    "PIPELINE_INFOS",
    "Pipeline",
    "PipelineInfo",
    "register_architecture",
    "PipelineInputConfig",
    "PipelineInputName",
    "extract_domain_inputs",
]
