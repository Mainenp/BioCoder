from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.schemas import ChatResponse


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ConversationStore:
    """Small SQLite store for durable local conversation history."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL REFERENCES conversations(thread_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    task_id TEXT,
                    plan_json TEXT NOT NULL DEFAULT '[]',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    tools_json TEXT NOT NULL DEFAULT '[]',
                    attachments_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id);
                CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "task_id" not in columns:
                connection.execute("ALTER TABLE messages ADD COLUMN task_id TEXT")
            if "attachments_json" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN attachments_json TEXT NOT NULL DEFAULT '[]'"
                )

            conversation_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "user_id" not in conversation_columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations(user_id, updated_at DESC)
                """
            )

    def save_turn(
        self,
        thread_id: str,
        question: str,
        response: ChatResponse,
        *,
        user_id: str,
    ) -> None:
        timestamp = _now()
        title = " ".join(question.strip().split())[:44] or "未命名研究"
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO conversations(thread_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at
                WHERE conversations.user_id = excluded.user_id
                """,
                (thread_id, user_id, title, timestamp, timestamp),
            )
            if cursor.rowcount == 0:
                raise PermissionError("Conversation belongs to another user")
            connection.execute(
                """
                INSERT INTO messages(thread_id, role, content, attachments_json, created_at)
                VALUES (?, 'user', ?, ?, ?)
                """,
                (
                    thread_id,
                    question,
                    json.dumps(
                        [item.model_dump(mode="json") for item in response.attachments],
                        ensure_ascii=False,
                    ),
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO messages(
                    thread_id, role, content, task_id, plan_json, sources_json, tools_json, created_at
                ) VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    response.answer,
                    response.task_id,
                    json.dumps(response.plan, ensure_ascii=False),
                    json.dumps([source.model_dump() for source in response.sources], ensure_ascii=False),
                    json.dumps(response.tools_used, ensure_ascii=False),
                    timestamp,
                ),
            )

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.thread_id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.thread_id = c.thread_id
                WHERE c.user_id = ?
                GROUP BY c.thread_id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (user_id, max(1, min(limit, 200))),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, thread_id: str, user_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE thread_id = ? AND user_id = ?",
                (thread_id, user_id),
            ).fetchone()
            if conversation is None:
                return None
            messages = connection.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY id", (thread_id,)
            ).fetchall()
        return {
            **dict(conversation),
            "messages": [self._deserialize_message(row) for row in messages],
        }

    def messages_for_agent(
        self, thread_id: str, user_id: str, limit: int = 16
    ) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.role, m.content FROM messages m
                JOIN conversations c ON c.thread_id = m.thread_id
                WHERE m.thread_id = ? AND c.user_id = ? ORDER BY m.id DESC LIMIT ?
                """,
                (thread_id, user_id, max(1, min(limit, 40))),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def owns_thread(self, thread_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversations WHERE thread_id = ? AND user_id = ?",
                (thread_id, user_id),
            ).fetchone()
        return row is not None

    def delete_conversation(self, thread_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE thread_id = ? AND user_id = ?",
                (thread_id, user_id),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _deserialize_message(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "role": row["role"],
            "content": row["content"],
            "task_id": row["task_id"],
            "plan": json.loads(row["plan_json"]),
            "sources": json.loads(row["sources_json"]),
            "tools": json.loads(row["tools_json"]),
            "attachments": json.loads(row["attachments_json"]),
            "created_at": row["created_at"],
        }
