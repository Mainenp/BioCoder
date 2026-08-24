from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eval.runner import run_benchmark, write_reports
from training.common import dataset_metadata, git_commit, load_yaml, new_run_dir, write_run_metadata
from training.sft.train import LoraSettings

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


class DPOTrainingSettings(BaseModel):
    learning_rate: float = Field(gt=0)
    epochs: float = Field(gt=0)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_length: int = Field(default=4096, ge=128)
    max_prompt_length: int = Field(default=2048, ge=64)
    beta: float = Field(default=0.1, gt=0)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    bf16: bool = True


class DPOJobConfig(BaseModel):
    model_name: str = Field(min_length=1)
    dataset_path: Path
    dataset_version: str = "unset"
    output_root: Path = Path("models")
    regression_dataset: Path
    regression_predictions: Path | None = None
    lora: LoraSettings = Field(default_factory=LoraSettings)
    training: DPOTrainingSettings


def load_config(path: Path, dataset_override: Path | None = None) -> DPOJobConfig:
    config = DPOJobConfig.model_validate(load_yaml(path))
    if dataset_override:
        config.dataset_path = dataset_override
    for field_name in ("dataset_path", "output_root", "regression_dataset", "regression_predictions"):
        value = getattr(config, field_name)
        if value is not None and not value.is_absolute():
            setattr(config, field_name, BACKEND_ROOT / value)
    return config


def _execute_training(config: DPOJobConfig, run_dir: Path) -> dict[str, Any]:
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Real DPO requires the optional training stack: transformers, datasets, trl, peft, accelerate."
        ) from exc

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = load_dataset("json", data_files=str(config.dataset_path), split="train")
    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules or None,
        task_type="CAUSAL_LM",
    )
    args = DPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_length=config.training.max_length,
        max_prompt_length=config.training.max_prompt_length,
        beta=config.training.beta,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        bf16=config.training.bf16,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    output = trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    return {"train_metrics": output.metrics, "checkpoint": str(run_dir / "adapter")}


def _regression(config: DPOJobConfig, run_dir: Path, model_version: str) -> dict[str, Any]:
    if config.regression_predictions is None:
        return {
            "status": "pending_candidate_predictions",
            "reason": "Generate candidate predictions, then rerun the regression gate before promotion.",
        }
    report = asyncio.run(
        run_benchmark(
            config.regression_dataset,
            predictions_path=config.regression_predictions,
            model_version=model_version,
        )
    )
    paths = write_reports(report, run_dir / "regression")
    return {"status": "completed", "report": str(paths["json"]), "metrics": report["metrics"]}


def run(config: DPOJobConfig, *, execute: bool = False) -> dict[str, Any]:
    dataset = dataset_metadata(config.dataset_path)
    run_dir = new_run_dir(config.output_root, "dpo")
    model_version = run_dir.name
    metadata: dict[str, Any] = {
        "training_method": "dpo_lora",
        "status": "running" if execute else "dry_run_validated",
        "dry_run": not execute,
        "model_name": config.model_name,
        "model_version": model_version,
        "dataset_version": config.dataset_version,
        "dataset": dataset,
        "training_config": config.model_dump(mode="json"),
        "git_commit": git_commit(BACKEND_ROOT.parent),
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": {},
        "checkpoint_metadata": {},
        "regression": {"status": "not_run_dry_run"},
    }
    write_run_metadata(run_dir, metadata)
    if execute:
        outcome = _execute_training(config, run_dir)
        metadata["status"] = "trained_unvalidated"
        metadata["metrics"] = outcome["train_metrics"]
        metadata["checkpoint_metadata"] = {"path": outcome["checkpoint"]}
        metadata["regression"] = _regression(config, run_dir, model_version)
        write_run_metadata(run_dir, metadata)
    metadata["run_dir"] = str(run_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or validate BioCoder LoRA DPO.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(load_config(args.config, args.dataset), execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
