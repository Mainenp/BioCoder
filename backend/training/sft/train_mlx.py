from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from training.common import (
    dataset_metadata,
    git_commit,
    load_yaml,
    new_run_dir,
    write_run_metadata,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.mlx.yaml")


class MLXLoraSettings(BaseModel):
    num_layers: int = Field(default=8, ge=1)
    rank: int = Field(default=8, ge=1)
    scale: float = Field(default=16.0, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    keys: list[str] = Field(
        default_factory=lambda: ["self_attn.q_proj", "self_attn.v_proj"]
    )


class MLXTrainingSettings(BaseModel):
    iterations: int = Field(default=300, ge=1)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    max_seq_length: int = Field(default=2048, ge=128)
    gradient_checkpointing: bool = True
    mask_prompt: bool = True
    logging_steps: int = Field(default=10, ge=1)
    evaluation_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=100, ge=1)
    clear_cache_threshold_bytes: int = Field(default=1024**3, ge=0)


class MLXSFTJobConfig(BaseModel):
    model_name: str = Field(min_length=1)
    dataset_path: Path
    dataset_version: str = "unset"
    output_root: Path = Path("models")
    seed: int = 42
    validation_fraction: float = Field(default=0.1, ge=0, lt=0.5)
    lora: MLXLoraSettings = Field(default_factory=MLXLoraSettings)
    training: MLXTrainingSettings = Field(default_factory=MLXTrainingSettings)


def load_config(path: Path, dataset_override: Path | None = None) -> MLXSFTJobConfig:
    config = MLXSFTJobConfig.model_validate(load_yaml(path))
    if dataset_override:
        config.dataset_path = dataset_override
    elif not config.dataset_path.is_absolute():
        config.dataset_path = BACKEND_ROOT / config.dataset_path
    if not config.output_root.is_absolute():
        config.output_root = BACKEND_ROOT / config.output_root
    return config


def _load_sft_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"SFT row {index} must contain at least two messages")
        if messages[-1].get("role") != "assistant" or not str(
            messages[-1].get("content") or ""
        ).strip():
            raise ValueError(f"SFT row {index} must end with a non-empty assistant answer")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def prepare_dataset(config: MLXSFTJobConfig, run_dir: Path) -> dict[str, Any]:
    rows = _load_sft_rows(config.dataset_path)
    shuffled = list(rows)
    random.Random(config.seed).shuffle(shuffled)
    validation_rows = 0
    if len(shuffled) >= 5 and config.validation_fraction > 0:
        validation_rows = max(1, round(len(shuffled) * config.validation_fraction))

    data_dir = run_dir / "mlx_data"
    data_dir.mkdir(parents=True, exist_ok=False)
    valid = shuffled[:validation_rows]
    train = shuffled[validation_rows:]
    _write_jsonl(data_dir / "train.jsonl", train)
    if valid:
        _write_jsonl(data_dir / "valid.jsonl", valid)
    return {
        "path": str(data_dir),
        "train_rows": len(train),
        "validation_rows": len(valid),
    }


def _mlx_config(config: MLXSFTJobConfig, run_dir: Path, data_dir: Path) -> dict[str, Any]:
    return {
        "model": config.model_name,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "data": str(data_dir),
        "seed": config.seed,
        "num_layers": config.lora.num_layers,
        "batch_size": config.training.batch_size,
        "iters": config.training.iterations,
        "val_batches": -1,
        "learning_rate": config.training.learning_rate,
        "steps_per_report": config.training.logging_steps,
        "steps_per_eval": config.training.evaluation_steps,
        "grad_accumulation_steps": config.training.gradient_accumulation_steps,
        "adapter_path": str(run_dir / "adapter"),
        "save_every": config.training.save_steps,
        "test": False,
        "max_seq_length": config.training.max_seq_length,
        "grad_checkpoint": config.training.gradient_checkpointing,
        "clear_cache_threshold": config.training.clear_cache_threshold_bytes,
        "mask_prompt": config.training.mask_prompt,
        "lora_parameters": {
            "keys": config.lora.keys,
            "rank": config.lora.rank,
            "scale": config.lora.scale,
            "dropout": config.lora.dropout,
        },
    }


def _validate_runtime() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX SFT execution requires an Apple Silicon Mac (Darwin arm64).")
    if importlib.util.find_spec("mlx_lm") is None:
        raise RuntimeError(
            "MLX-LM is not installed. Run `make setup-training-mac` from the project root."
        )


def _execute_training(effective_config_path: Path, adapter_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(effective_config_path),
    ]
    subprocess.run(command, cwd=BACKEND_ROOT, check=True)
    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.exists():
        raise RuntimeError(f"MLX-LM completed without producing {adapter_file}")
    return {"path": str(adapter_path), "adapter_file": str(adapter_file), "command": command}


def run(config: MLXSFTJobConfig, *, execute: bool = False) -> dict[str, Any]:
    dataset = dataset_metadata(config.dataset_path)
    if execute:
        _validate_runtime()
    run_dir = new_run_dir(config.output_root, "sft_mlx")
    prepared_dataset = prepare_dataset(config, run_dir)
    effective_config = _mlx_config(config, run_dir, Path(prepared_dataset["path"]))
    effective_config_path = run_dir / "mlx_lora_config.yaml"
    effective_config_path.write_text(
        yaml.safe_dump(effective_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "training_method": "sft_qlora_mlx",
        "status": "running" if execute else "dry_run_validated",
        "dry_run": not execute,
        "model_name": config.model_name,
        "dataset_version": config.dataset_version,
        "dataset": dataset,
        "prepared_dataset": prepared_dataset,
        "training_config": config.model_dump(mode="json"),
        "effective_mlx_config": str(effective_config_path),
        "hardware_profile": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "target_unified_memory_gb": 16,
        },
        "git_commit": git_commit(BACKEND_ROOT.parent),
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": {},
        "checkpoint_metadata": {},
    }
    write_run_metadata(run_dir, metadata)
    if execute:
        try:
            outcome = _execute_training(effective_config_path, run_dir / "adapter")
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = str(exc)
            metadata["completed_at"] = datetime.now(UTC).isoformat()
            write_run_metadata(run_dir, metadata)
            raise
        metadata["status"] = "trained_unvalidated"
        metadata["checkpoint_metadata"] = outcome
        metadata["completed_at"] = datetime.now(UTC).isoformat()
        write_run_metadata(run_dir, metadata)
    metadata["run_dir"] = str(run_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or validate BioCoder MLX QLoRA SFT.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(load_config(args.config, args.dataset), execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
