from __future__ import annotations

import re
from pathlib import Path

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret)\s*[:=]\s*[^\s,;]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"),
)

INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(?:all\s+)?previous\s+instructions"),
    re.compile(r"(?i)reveal\s+(?:the\s+)?(?:system\s+)?prompt"),
    re.compile(r"(?i)developer\s+message"),
    re.compile(r"(?i)execute\s+(?:this\s+)?(?:shell|command)"),
)

UNSAFE_SHELL_PATTERNS = (
    re.compile(r"(?:^|\s)(?:rm\s+-rf|sudo|chmod\s+777|curl\s+[^|]+\|\s*(?:sh|bash))(?:\s|$)"),
    re.compile(r"(?:;|&&|\|\|)\s*(?:rm|sudo|sh|bash)\b"),
)

MODEL_PROTOCOL_PATTERNS = (
    re.compile(r"<[^>\n]*DSML[^>\n]*(?:tool_calls|invoke|parameter)\b", re.IGNORECASE),
    re.compile(r"<\/?\s*(?:tool_calls?|function_calls?|invoke)\b", re.IGNORECASE),
)


def redact_secrets(value: str) -> str:
    result = value
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def contains_model_protocol_artifact(value: str) -> bool:
    """Detect model-internal tool protocol that must never be shown or trained on."""
    return any(pattern.search(value) for pattern in MODEL_PROTOCOL_PATTERNS)


def detect_prompt_injection(value: str) -> list[str]:
    return [pattern.pattern for pattern in INJECTION_PATTERNS if pattern.search(value)]


def validate_tool_text(
    value: str,
    *,
    max_length: int = 2000,
    reject_prompt_injection: bool = True,
) -> str:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise ValueError("Tool text input must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"Tool text input exceeds {max_length} characters")
    if "\x00" in value:
        raise ValueError("Tool text input contains a null byte")
    if any(pattern.search(normalized) for pattern in SECRET_PATTERNS):
        raise ValueError("Tool text input appears to contain a secret")
    if reject_prompt_injection and detect_prompt_injection(normalized):
        raise ValueError("Potential prompt/tool injection detected")
    return normalized


def validate_shell_command(command: str) -> str:
    normalized = validate_tool_text(command, max_length=4000, reject_prompt_injection=True)
    if any(pattern.search(normalized) for pattern in UNSAFE_SHELL_PATTERNS):
        raise ValueError("Unsafe shell command rejected")
    return normalized


def ensure_path_within(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Path escapes allowed root: {path}")
    return resolved
