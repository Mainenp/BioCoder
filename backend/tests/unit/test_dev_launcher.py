import importlib.util
from pathlib import Path

import pytest

DEV_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "dev.py"
SPEC = importlib.util.spec_from_file_location("bioagent_dev_launcher", DEV_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dev)


def test_read_dotenv_handles_comments_and_quotes(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\nPLAIN=value\nDOUBLE=\"two words\"\nSINGLE='three words'\n",
        encoding="utf-8",
    )

    assert dev.read_dotenv(env_path) == {
        "PLAIN": "value",
        "DOUBLE": "two words",
        "SINGLE": "three words",
    }


def test_local_mlx_command_uses_project_relative_model(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "backend/models/base/Qwen3-4B-Instruct-2507-4bit"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    python_path = tmp_path / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    (tmp_path / ".env").write_text(
        "MLX_AUTO_START=true\n"
        "MLX_MODEL_PATH=backend/models/base/Qwen3-4B-Instruct-2507-4bit\n"
        "MLX_PORT=18080\n"
        "OPENAI_BASE_URL=http://127.0.0.1:18080/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dev, "ROOT", tmp_path)

    command, message = dev.local_mlx_command()

    assert command is not None
    assert command[:4] == [str(python_path), "-m", "mlx_lm", "server"]
    assert command[command.index("--model") + 1] == (
        "backend/models/base/Qwen3-4B-Instruct-2507-4bit"
    )
    assert "18080" in (message or "")


def test_local_mlx_command_supports_vision_server(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "backend/models/base/Qwen3-VL-4B-Instruct-4bit"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
    python_path = tmp_path / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    (tmp_path / ".env").write_text(
        "MLX_AUTO_START=true\n"
        "MLX_SERVER_MODULE=mlx_vlm.server\n"
        "MLX_MODEL_PATH=backend/models/base/Qwen3-VL-4B-Instruct-4bit\n"
        "MLX_PORT=18080\n"
        "OPENAI_BASE_URL=http://127.0.0.1:18080/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dev, "ROOT", tmp_path)

    command, message = dev.local_mlx_command()

    assert command is not None
    assert command[:3] == [str(python_path), "-m", "mlx_vlm.server"]
    assert command[command.index("--model") + 1] == "backend/models/base/Qwen3-VL-4B-Instruct-4bit"
    assert "MLX-VLM" in (message or "")


def test_local_mlx_command_rejects_non_loopback_endpoint(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "MLX_AUTO_START=true\nOPENAI_BASE_URL=https://example.com/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dev, "ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="loopback"):
        dev.local_mlx_command()
