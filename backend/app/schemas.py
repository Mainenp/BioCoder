from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _normalize_email(value: str) -> str:
    normalized = value.strip().lower()
    if (
        len(normalized) > 254
        or normalized.count("@") != 1
        or "." not in normalized.rsplit("@", 1)[1]
        or any(character.isspace() for character in normalized)
    ):
        raise ValueError("请输入有效的邮箱地址。")
    return normalized


class AuthUser(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: str


class AuthConfig(BaseModel):
    registration_enabled: bool
    invite_required: bool
    admin_email: str


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _normalize_email(value)


class RegisterRequest(LoginRequest):
    display_name: str = Field(min_length=2, max_length=40)
    invite_code: str = Field(min_length=1, max_length=128)

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if len(normalized) < 2:
            raise ValueError("昵称至少需要 2 个字符。")
        return normalized


class Attachment(BaseModel):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    name: str = Field(min_length=1, max_length=160)
    kind: Literal["image", "pdf", "word", "text"]
    media_type: str
    size_bytes: int = Field(ge=1)
    extracted_characters: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = Field(default=None, max_length=100)
    attachment_ids: list[str] = Field(default_factory=list, max_length=4)


class Source(BaseModel):
    title: str
    url: str | None = None
    source_type: str = "knowledge_base"
    snippet: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    task_id: str | None = None
    trace_id: str | None = None
    plan: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)


class KnowledgeStatus(BaseModel):
    ready: bool
    documents: int
    chunks: int
    files: list[str]


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    knowledge_ready: bool
    attachments_enabled: bool = False
    vision_input_enabled: bool = False
    attachment_formats: list[str] = Field(default_factory=list)
    attachment_max_files: int = 0
    attachment_max_file_bytes: int = 0
    version: str = "2.0.0"


class TrainingTriggerRequest(BaseModel):
    execute: bool = False


class ConversationSummary(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationMessage(BaseModel):
    id: str
    role: str
    content: str
    task_id: str | None = None
    plan: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    created_at: str


class ConversationDetail(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ConversationMessage] = Field(default_factory=list)
