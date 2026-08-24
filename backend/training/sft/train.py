from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from training.common import (
    dataset_metadata,
    git_commit,
    load_yaml,
    new_run_dir,
    write_run_metadata,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


class LoraSettings(BaseModel):
    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=list)


class TrainingSettings(BaseModel):
    learning_rate: float = Field(gt=0)
    epochs: float = Field(gt=0)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_seq_length: int = Field(default=4096, ge=128)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    bf16: bool = True


class SFTJobConfig(BaseModel):
    model_name: str = Field(min_length=1)
    dataset_path: Path
    dataset_version: str = "unset"
    output_root: Path = Path("models")
    lora: LoraSettings = Field(default_factory=LoraSettings)
    training: TrainingSettings


def load_config(path: Path, dataset_override: Path | None = None) -> SFTJobConfig:
    config = SFTJobConfig.model_validate(load_yaml(path))
    if dataset_override:
        config.dataset_path = dataset_override
    elif not config.dataset_path.is_absolute():
        config.dataset_path = BACKEND_ROOT / config.dataset_path
    if not config.output_root.is_absolute():
        config.output_root = BACKEND_ROOT / config.output_root
    return config


def _execute_training(config: SFTJobConfig, run_dir: Path) -> dict[str, Any]:
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Real SFT requires the optional training stack: transformers, datasets, trl, peft, accelerate."
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
    args = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_length=config.training.max_seq_length,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        bf16=config.training.bf16,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    output = trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    return {"train_metrics": output.metrics, "checkpoint": str(run_dir / "adapter")}


def run(config: SFTJobConfig, *, execute: bool = False) -> dict[str, Any]:
    dataset = dataset_metadata(config.dataset_path)
    run_dir = new_run_dir(config.output_root, "sft")
    metadata: dict[str, Any] = {
        "training_method": "sft_lora",
        "status": "running" if execute else "dry_run_validated",
        "dry_run": not execute,
        "model_name": config.model_name,
        "dataset_version": config.dataset_version,
        "dataset": dataset,
        "training_config": config.model_dump(mode="json"),
        "git_commit": git_commit(BACKEND_ROOT.parent),
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": {},
        "checkpoint_metadata": {},
    }
    write_run_metadata(run_dir, metadata)
    if execute:
        outcome = _execute_training(config, run_dir)
        metadata["status"] = "trained_unvalidated"
        metadata["metrics"] = outcome["train_metrics"]
        metadata["checkpoint_metadata"] = {"path": outcome["checkpoint"]}
        write_run_metadata(run_dir, metadata)
    metadata["run_dir"] = str(run_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or validate BioCoder LoRA SFT.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--execute", action="store_true", help="Run training instead of a dry-run validation.")
    args = parser.parse_args()
    result = run(load_config(args.config, args.dataset), execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
