from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_origin: str = "http://localhost:5173"

    auth_registration_enabled: bool = True
    auth_invite_code: str = ""
    auth_admin_email: str = ""
    auth_session_days: int = 30
    auth_cookie_name: str = "biocoder_session"

    openai_api_key: str = Field(default="")
    openai_base_url: str = "https://api.openai.com/v1"
    model_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_thinking_mode: str = "disabled"
    llm_max_tokens: int = 4096
    llm_max_retries: int = 1
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "auto"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    vllm_base_url: str = ""
    vllm_model: str = ""
    vllm_api_key: str = ""
    vllm_embedding_model: str = ""

    enable_web_tools: bool = True
    max_tool_rounds: int = 6
    request_timeout_seconds: float = 20.0
    agent_timeout_seconds: float = 90.0
    max_steps: int = 20
    max_retries: int = 3
    token_budget: int = 100000
    cost_budget: float = 10.0
    ncbi_api_key: str = ""
    ncbi_email: str = ""

    agent_version: str = "2.0.0"
    prompt_version: str = "v1"
    model_version: str = ""
    prompt_cost_per_million: float = 0.0
    completion_cost_per_million: float = 0.0
    bad_case_score_threshold: float = 0.65

    knowledge_dir: Path = BACKEND_ROOT / "data" / "knowledge"
    uploads_dir: Path = BACKEND_ROOT / "data" / "uploads"
    attachments_dir: Path = BACKEND_ROOT / "data" / "attachments"
    attachments_enabled: bool = True
    vision_input_enabled: bool = False
    attachment_max_file_bytes: int = 15 * 1024 * 1024
    attachment_max_files: int = 4
    attachment_max_text_characters: int = 60_000
    attachment_max_vision_images: int = 6
    attachment_max_visual_bytes: int = 20 * 1024 * 1024
    attachment_pdf_vision_pages: int = 4
    knowledge_exclude_files: str = "demo_knowledge.md"
    history_db_path: Path = BACKEND_ROOT / "data" / "bioagent.db"
    trajectory_dir: Path = BACKEND_ROOT / "data" / "trajectories"
    trajectory_jsonl_path: Path = BACKEND_ROOT / "data" / "trajectories.jsonl"
    feedback_dir: Path = BACKEND_ROOT / "data" / "feedback"
    bad_case_dir: Path = BACKEND_ROOT / "data" / "bad_cases"
    dataset_dir: Path = BACKEND_ROOT / "data" / "datasets"
    model_registry_path: Path = BACKEND_ROOT / "data" / "model_registry.json"
    semantic_memory_dir: Path = BACKEND_ROOT / "data" / "memory" / "semantic"
    memory_minimum_quality: float = 0.75
    evaluation_results_dir: Path = BACKEND_ROOT / "results"

    training_api_enabled: bool = False
    training_api_allow_execute: bool = False
    auto_sft_enabled: bool = False
    auto_sft_execute: bool = False
    auto_sft_min_new_positive_feedback: int = 20
    auto_sft_min_records: int = 20
    auto_sft_cooldown_seconds: int = 86400
    auto_sft_config_path: Path = BACKEND_ROOT / "training" / "sft" / "config.mlx.yaml"
    training_job_state_path: Path = BACKEND_ROOT / "results" / "training_job.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
