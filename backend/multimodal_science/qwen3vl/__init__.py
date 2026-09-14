"""Qwen3-VL instruction data built from the verified ChromPeakFormer Dataset."""

from multimodal_science.qwen3vl.instruction_data import (
    InstructionDatasetResult,
    build_instruction_dataset,
)
from multimodal_science.qwen3vl.evaluation import (
    QwenEvaluationResult,
    evaluate_qwen_predictions,
)
from multimodal_science.qwen3vl.inference import (
    AdapterSpec,
    GenerationSettings,
    QwenInferenceResult,
    run_qwen_inference,
)
from multimodal_science.qwen3vl.inference_bundle import (
    InferenceBundleResult,
    build_inference_bundle,
)
from multimodal_science.qwen3vl.zero_shot_audit import (
    ZeroShotAuditResult,
    audit_zero_shot_failures,
)
from multimodal_science.qwen3vl.lora_data import (
    LoraBundleResult,
    build_lora_training_bundle,
)
from multimodal_science.qwen3vl.lora_training import (
    LoraTrainingResult,
    LoraTrainingSettings,
    run_lora_training,
)
from multimodal_science.qwen3vl.development_comparison import (
    CrossFamilyComparisonResult,
    build_cross_family_development_comparison,
)
from multimodal_science.qwen3vl.fusion_data import (
    FusionBundleResult,
    build_fusion_bundle,
)
from multimodal_science.qwen3vl.sensor_projector import (
    SensorProjectorSpec,
    build_sensor_projector,
)

__all__ = [
    "AdapterSpec",
    "InstructionDatasetResult",
    "InferenceBundleResult",
    "LoraBundleResult",
    "LoraTrainingResult",
    "LoraTrainingSettings",
    "QwenEvaluationResult",
    "QwenInferenceResult",
    "ZeroShotAuditResult",
    "GenerationSettings",
    "CrossFamilyComparisonResult",
    "FusionBundleResult",
    "SensorProjectorSpec",
    "build_fusion_bundle",
    "build_inference_bundle",
    "build_instruction_dataset",
    "build_lora_training_bundle",
    "build_cross_family_development_comparison",
    "evaluate_qwen_predictions",
    "audit_zero_shot_failures",
    "run_qwen_inference",
    "run_lora_training",
    "build_sensor_projector",
]
