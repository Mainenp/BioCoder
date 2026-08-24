from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

import app.main as main_module
from app.services.attachments import AttachmentStore
from app.services.auth import AuthStore
from app.services.history import ConversationStore
from biocoder.bad_cases.store import BadCaseStore
from biocoder.memory.store import SemanticMemoryStore
from biocoder.trajectory.storage import TrajectoryStorage
from feedback.store import FeedbackStore

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8"
    b"\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeAgent:
    def __init__(self) -> None:
        self.attachments = []

    async def ask(self, message, history=None, memory_context=None, attachments=None):
        self.attachments = attachments or []
        return {
            "messages": [HumanMessage(content=message), AIMessage(content="Evidence-based answer")],
            "plan": ["retrieve", "verify", "answer"],
            "tool_rounds": 0,
        }


def test_chat_saves_trajectory_and_negative_feedback_creates_bad_case(tmp_path, monkeypatch) -> None:
    trajectories = TrajectoryStorage(tmp_path / "trajectories", tmp_path / "trajectories.jsonl")
    bad_cases = BadCaseStore(tmp_path / "bad_cases")
    monkeypatch.setattr(main_module, "conversation_store", ConversationStore(tmp_path / "history.db"))
    monkeypatch.setattr(main_module, "auth_store", AuthStore(tmp_path / "auth.db"))
    monkeypatch.setattr(main_module.settings, "auth_invite_code", "test-invite")
    monkeypatch.setattr(main_module, "trajectory_storage", trajectories)
    monkeypatch.setattr(main_module, "feedback_store", FeedbackStore(tmp_path / "feedback"))
    monkeypatch.setattr(main_module, "bad_case_store", bad_cases)
    memory_store = SemanticMemoryStore(tmp_path / "memory", minimum_write_quality=0.75)
    monkeypatch.setattr(main_module, "semantic_memory_store", memory_store)
    test_settings = main_module.settings.model_copy(
        update={"attachments_dir": tmp_path / "attachments", "vision_input_enabled": True}
    )
    monkeypatch.setattr(main_module, "attachment_store", AttachmentStore(test_settings))
    fake_agent = FakeAgent()
    monkeypatch.setattr(main_module, "get_agent", lambda: fake_agent)

    with TestClient(main_module.app) as client:
        registration = client.post(
            "/api/auth/register",
            json={
                "display_name": "E2E Researcher",
                "email": "e2e@example.com",
                "password": "correct-horse-battery",
                "invite_code": "test-invite",
            },
        )
        assert registration.status_code == 201
        upload = client.post(
            "/api/attachments",
            files={"file": ("cell.png", PNG_1X1, "image/png")},
        )
        assert upload.status_code == 201
        chat = client.post(
            "/api/chat",
            json={"message": "Explain EGFR", "attachment_ids": [upload.json()["id"]]},
        )
        assert chat.status_code == 200
        payload = chat.json()
        assert payload["task_id"]
        assert payload["trace_id"]
        assert payload["attachments"][0]["name"] == "cell.png"
        assert payload["sources"][0]["source_type"] == "attachment_image"
        assert payload["tools_used"][0] == "read_attachment"
        assert fake_agent.attachments[0].descriptor.name == "cell.png"
        trajectory = trajectories.load(payload["task_id"])
        assert trajectory is not None
        assert trajectory.final_answer == "Evidence-based answer"

        feedback = client.post(
            "/api/feedback",
            json={"task_id": payload["task_id"], "feedback_type": "thumbs_down"},
        )
        assert feedback.status_code == 200
        assert feedback.json()["task_id"] == payload["task_id"]
        assert len(bad_cases.all()) == 1
        task_memories = [
            record for record in memory_store.all() if record.source_task == payload["task_id"]
        ]
        assert task_memories
        assert all(record.owner_id == registration.json()["id"] for record in task_memories)
        assert all(record.active is False for record in task_memories)

        duplicate = client.post(
            "/api/feedback",
            json={"task_id": payload["task_id"], "feedback_type": "rating", "rating": 1},
        )
        assert duplicate.status_code == 409
