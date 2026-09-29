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
from multimodal_science.qwen3vl.sensor_fusion import insert_sensor_embeddings
from multimodal_science.qwen3vl.fusion_training import (
    FusionTrainingResult,
    FusionTrainingSettings,
    run_fusion_training,
)
from multimodal_science.qwen3vl.fusion_inference import run_fusion_inference
from multimodal_science.qwen3vl.auxiliary_pretraining_data import (
    AuxiliaryPretrainingDatasetResult,
    build_auxiliary_pretraining_dataset,
)
from multimodal_science.qwen3vl.auxiliary_pretraining import (
    AuxiliaryPretrainingResult,
    AuxiliaryPretrainingSettings,
    run_auxiliary_pretraining,
)

__all__ = [
    "AdapterSpec",
    "AuxiliaryPretrainingDatasetResult",
    "AuxiliaryPretrainingResult",
    "AuxiliaryPretrainingSettings",
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
    "FusionTrainingResult",
    "FusionTrainingSettings",
    "SensorProjectorSpec",
    "build_fusion_bundle",
    "build_auxiliary_pretraining_dataset",
    "build_inference_bundle",
    "build_instruction_dataset",
    "build_lora_training_bundle",
    "build_cross_family_development_comparison",
    "evaluate_qwen_predictions",
    "audit_zero_shot_failures",
    "run_qwen_inference",
    "run_lora_training",
    "build_sensor_projector",
    "insert_sensor_embeddings",
    "run_fusion_training",
    "run_fusion_inference",
    "run_auxiliary_pretraining",
]
