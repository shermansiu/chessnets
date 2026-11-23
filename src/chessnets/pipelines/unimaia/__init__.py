"""UniMaia inference pipeline."""

from chessnets.utils.time_control import TimeInfoPromptStyle

from .pipeline_unimaia import (
    END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE,
    PipelineInputConfig,
    UniMaiaDataset,
    UniMaiaDatasetWithClockInfo,
    UniMaiaDatasetWithTimeControl,
    UniMaiaInputMode,
    UniMaiaPipeline,
    UniMaiaPipelineConfig,
    all_termination_reasons,
)

__all__ = [
    "END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE",
    "PipelineInputConfig",
    "TimeInfoPromptStyle",
    "UniMaiaDataset",
    "UniMaiaDatasetWithClockInfo",
    "UniMaiaDatasetWithTimeControl",
    "UniMaiaInputMode",
    "UniMaiaPipeline",
    "UniMaiaPipelineConfig",
    "all_termination_reasons",
]
