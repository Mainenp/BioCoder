from app.schemas import Attachment, ChatResponse, Source
from app.services.history import ConversationStore


def test_conversation_store_persists_and_restores_turns(tmp_path) -> None:
    store = ConversationStore(tmp_path / "history.db")
    store.initialize()
    response = ChatResponse(
        thread_id="thread-1",
        task_id="task-1",
        answer="C797S 会影响共价结合。",
        plan=["检索机制", "核对证据"],
        sources=[Source(title="Paper", url="https://example.com", source_type="pubmed")],
        tools_used=["search_pubmed"],
        attachments=[
            Attachment(
                id="a" * 32,
                name="evidence.pdf",
                kind="pdf",
                media_type="application/pdf",
                size_bytes=128,
            )
        ],
    )

    store.save_turn("thread-1", "分析 EGFR C797S 耐药机制", response, user_id="user-1")
    store.save_turn("thread-1", "有哪些潜在策略？", response, user_id="user-1")

    summaries = store.list_conversations("user-1")
    assert len(summaries) == 1
    assert summaries[0]["message_count"] == 4
    assert summaries[0]["title"] == "分析 EGFR C797S 耐药机制"

    assert store.list_conversations("user-2") == []
    assert store.get_conversation("thread-1", "user-2") is None

    detail = store.get_conversation("thread-1", "user-1")
    assert detail is not None
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert detail["messages"][1]["sources"][0]["title"] == "Paper"
    assert detail["messages"][1]["task_id"] == "task-1"
    assert detail["messages"][0]["task_id"] is None
    assert detail["messages"][0]["attachments"][0]["name"] == "evidence.pdf"
    assert detail["messages"][1]["attachments"] == []
    assert len(store.messages_for_agent("thread-1", "user-1")) == 4
    assert store.messages_for_agent("thread-1", "user-2") == []

    assert store.delete_conversation("thread-1", "user-2") is False
    assert store.delete_conversation("thread-1", "user-1") is True
    assert store.get_conversation("thread-1", "user-1") is None


def test_conversation_store_migrates_existing_messages_table(tmp_path) -> None:
    import sqlite3

    database = tmp_path / "legacy.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL REFERENCES conversations(thread_id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                plan_json TEXT NOT NULL DEFAULT '[]',
                sources_json TEXT NOT NULL DEFAULT '[]',
                tools_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            """
        )

    store = ConversationStore(database)
    store.initialize()
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
        conversation_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(conversations)")
        }
    assert "task_id" in columns
    assert "attachments_json" in columns
    assert "user_id" in conversation_columns
