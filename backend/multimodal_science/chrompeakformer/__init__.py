"""Execution boundary for the public ChromPeakFormer extractor contract."""

from multimodal_science.chrompeakformer.asset_index import (
    AssetIndexResult,
    build_asset_index,
)
from multimodal_science.chrompeakformer.executor import (
    ExecutionResult,
    execute_plan,
    run_job,
)
from multimodal_science.chrompeakformer.outputs import OutputSummary, validate_outputs
from multimodal_science.chrompeakformer.multimodal_dataset import (
    MultimodalDatasetResult,
    build_multimodal_dataset,
)
from multimodal_science.chrompeakformer.sequence_preflight import (
    SequencePreflightResult,
    build_sequence_preflight_report,
)
from multimodal_science.chrompeakformer.training_readiness import (
    TrainingReadinessResult,
    build_training_readiness_report,
)

__all__ = [
    "AssetIndexResult",
    "ExecutionResult",
    "MultimodalDatasetResult",
    "OutputSummary",
    "SequencePreflightResult",
    "TrainingReadinessResult",
    "build_asset_index",
    "build_multimodal_dataset",
    "build_training_readiness_report",
    "build_sequence_preflight_report",
    "execute_plan",
    "run_job",
    "validate_outputs",
]
