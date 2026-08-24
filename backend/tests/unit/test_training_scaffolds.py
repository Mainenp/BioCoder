import json
from pathlib import Path

from training.dpo.train import load_config as load_dpo_config
from training.dpo.train import run as run_dpo
from training.sft.train import load_config as load_sft_config
from training.sft.train import run as run_sft
from training.sft.train_mlx import load_config as load_mlx_sft_config
from training.sft.train_mlx import run as run_mlx_sft


def write_dataset(path, row) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_sft_dry_run_validates_data_and_writes_metadata(tmp_path) -> None:
    dataset = tmp_path / "sft.jsonl"
    write_dataset(dataset, {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        f"""
model_name: local/test-model
dataset_path: {dataset}
dataset_version: v1
output_root: {tmp_path / 'models'}
lora: {{r: 4, alpha: 8, dropout: 0.05}}
training:
  learning_rate: 0.0002
  epochs: 1
  per_device_train_batch_size: 1
  max_seq_length: 512
""",
        encoding="utf-8",
    )

    result = run_sft(load_sft_config(config_path))
    metadata = json.loads((tmp_path / "models" / "sft" / result["run_dir"].split("/")[-1] / "run_metadata.json").read_text())

    assert result["status"] == "dry_run_validated"
    assert result["dataset"]["rows"] == 1
    assert metadata["training_method"] == "sft_lora"
    assert metadata["checkpoint_metadata"] == {}


def test_dpo_dry_run_does_not_claim_regression_or_training(tmp_path) -> None:
    dataset = tmp_path / "preference.jsonl"
    write_dataset(dataset, {"prompt": "q", "chosen": "good", "rejected": "bad"})
    benchmark = tmp_path / "benchmark.jsonl"
    write_dataset(
        benchmark,
        {"id": "1", "task_type": "medical_qa", "query": "q", "expected": {}, "required_tools": [], "rubric": {}},
    )
    config_path = tmp_path / "dpo.yaml"
    config_path.write_text(
        f"""
model_name: local/test-model
dataset_path: {dataset}
dataset_version: v1
output_root: {tmp_path / 'models'}
regression_dataset: {benchmark}
regression_predictions: null
lora: {{r: 4, alpha: 8, dropout: 0.05}}
training:
  learning_rate: 0.000005
  epochs: 1
  per_device_train_batch_size: 1
  max_length: 512
  max_prompt_length: 256
""",
        encoding="utf-8",
    )

    result = run_dpo(load_dpo_config(config_path))

    assert result["status"] == "dry_run_validated"
    assert result["regression"]["status"] == "not_run_dry_run"
    assert result["checkpoint_metadata"] == {}


def test_mlx_sft_dry_run_prepares_deterministic_chat_dataset(tmp_path) -> None:
    dataset = tmp_path / "sft.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"question-{index}"},
                        {"role": "assistant", "content": f"answer-{index}"},
                    ]
                }
            )
            + "\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "mlx.yaml"
    config_path.write_text(
        f"""
model_name: mlx-community/Qwen3-0.6B-4bit
dataset_path: {dataset}
dataset_version: v1
output_root: {tmp_path / 'models'}
seed: 7
validation_fraction: 0.2
lora:
  num_layers: 4
  rank: 4
  scale: 8
training:
  iterations: 2
  batch_size: 1
  max_seq_length: 512
""",
        encoding="utf-8",
    )

    result = run_mlx_sft(load_mlx_sft_config(config_path))
    run_dir = tmp_path / "models" / "sft_mlx" / Path(result["run_dir"]).name

    assert result["status"] == "dry_run_validated"
    assert result["prepared_dataset"]["train_rows"] == 8
    assert result["prepared_dataset"]["validation_rows"] == 2
    assert (run_dir / "mlx_data" / "train.jsonl").exists()
    assert (run_dir / "mlx_data" / "valid.jsonl").exists()
    effective = (run_dir / "mlx_lora_config.yaml").read_text(encoding="utf-8")
    assert "fine_tune_type: lora" in effective
    assert "mask_prompt: true" in effective
