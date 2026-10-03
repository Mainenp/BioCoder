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
    FinalBenchmarkAccessSpec,
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
from multimodal_science.qwen3vl.xic_intervention_analysis import (
    XicInterventionAnalysisResult,
    XicInterventionRun,
    analyze_xic_interventions,
)
from multimodal_science.qwen3vl.fusion_matrix_analysis import (
    FusionMatrixAnalysisResult,
    FusionMatrixRun,
    analyze_fusion_matrix,
)
from multimodal_science.qwen3vl.development_dossier import (
    DevelopmentDossierResult,
    build_development_dossier,
)
from multimodal_science.qwen3vl.final_benchmark_protocol import (
    FinalBenchmarkAccessResult,
    FinalBenchmarkProtocolResult,
    FinalBenchmarkRuntimeContext,
    complete_final_benchmark_access,
    freeze_final_benchmark_protocol,
    load_final_benchmark_context,
    open_final_benchmark_access,
    verify_final_benchmark_access,
)
from multimodal_science.qwen3vl.final_benchmark_data import (
    FINAL_ANSWER_REPORT_SCHEMA,
    FINAL_BENCHMARK_DATA_SCHEMA,
    FINAL_DETECTOR_DATASET_SCHEMA,
    FINAL_FUSION_BUNDLE_SCHEMA,
    FINAL_INFERENCE_BUNDLE_SCHEMA,
    FinalBenchmarkDataResult,
    build_final_benchmark_data,
)
from multimodal_science.qwen3vl.final_benchmark_evaluation import (
    FINAL_QWEN_EVALUATION_SCHEMA,
    FinalQwenEvaluationResult,
    evaluate_final_qwen_predictions,
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
    "XicInterventionAnalysisResult",
    "XicInterventionRun",
    "GenerationSettings",
    "CrossFamilyComparisonResult",
    "DevelopmentDossierResult",
    "FusionBundleResult",
    "FusionMatrixAnalysisResult",
    "FusionMatrixRun",
    "FusionTrainingResult",
    "FusionTrainingSettings",
    "FinalBenchmarkAccessResult",
    "FinalBenchmarkAccessSpec",
    "FinalBenchmarkProtocolResult",
    "FinalBenchmarkRuntimeContext",
    "SensorProjectorSpec",
    "build_fusion_bundle",
    "build_auxiliary_pretraining_dataset",
    "build_inference_bundle",
    "build_instruction_dataset",
    "build_lora_training_bundle",
    "build_cross_family_development_comparison",
    "build_development_dossier",
    "complete_final_benchmark_access",
    "build_final_benchmark_data",
    "evaluate_qwen_predictions",
    "evaluate_final_qwen_predictions",
    "freeze_final_benchmark_protocol",
    "FinalBenchmarkDataResult",
    "FinalQwenEvaluationResult",
    "FINAL_ANSWER_REPORT_SCHEMA",
    "FINAL_BENCHMARK_DATA_SCHEMA",
    "FINAL_DETECTOR_DATASET_SCHEMA",
    "FINAL_FUSION_BUNDLE_SCHEMA",
    "FINAL_INFERENCE_BUNDLE_SCHEMA",
    "FINAL_QWEN_EVALUATION_SCHEMA",
    "audit_zero_shot_failures",
    "analyze_xic_interventions",
    "analyze_fusion_matrix",
    "run_qwen_inference",
    "open_final_benchmark_access",
    "load_final_benchmark_context",
    "run_lora_training",
    "build_sensor_projector",
    "insert_sensor_embeddings",
    "run_fusion_training",
    "run_fusion_inference",
    "run_auxiliary_pretraining",
    "verify_final_benchmark_access",
]
