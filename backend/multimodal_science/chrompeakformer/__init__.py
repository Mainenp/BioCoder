"""Execution boundary for the public ChromPeakFormer extractor contract."""

from multimodal_science.chrompeakformer.executor import (
    ExecutionResult,
    execute_plan,
    run_job,
)
from multimodal_science.chrompeakformer.outputs import OutputSummary, validate_outputs

__all__ = [
    "ExecutionResult",
    "OutputSummary",
    "execute_plan",
    "run_job",
    "validate_outputs",
]
