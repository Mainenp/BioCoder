"""Leakage-resistant baselines and metrics for scientific multimodal evaluation."""

from multimodal_science.baselines.comparison import (
    DevelopmentComparisonResult,
    build_development_comparison,
)
from multimodal_science.baselines.dataset import SequenceSplit, load_sequence_split
from multimodal_science.baselines.metrics import (
    binary_metrics,
    boundary_metrics,
    grouped_bootstrap_binary,
    select_threshold,
)
from multimodal_science.baselines.sequence_model import SequenceModelSpec
from multimodal_science.baselines.run_validation import (
    SequenceRunVerification,
    validate_sequence_run,
)
from multimodal_science.baselines.sequence_training import SequenceTrainConfig

__all__ = [
    "DevelopmentComparisonResult",
    "SequenceSplit",
    "SequenceModelSpec",
    "SequenceRunVerification",
    "SequenceTrainConfig",
    "binary_metrics",
    "boundary_metrics",
    "build_development_comparison",
    "grouped_bootstrap_binary",
    "load_sequence_split",
    "select_threshold",
    "validate_sequence_run",
]
