from fastapi.testclient import TestClient

import app.main as main_module
from app.rag.store import KnowledgeStore
from app.services.attachments import AttachmentStore
from app.services.auth import AuthStore

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8"
    b"\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)

app = main_module.app


def test_health_is_available_without_api_key() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "llm_configured" in response.json()
    assert response.json()["llm_configured"] is True
    assert response.json()["attachments_enabled"] is True
    assert ".docx" in response.json()["attachment_formats"]


def test_private_api_requires_login() -> None:
    with TestClient(app) as client:
        response = client.get("/api/conversations")
    assert response.status_code == 401


def test_register_me_and_logout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "auth_store", AuthStore(tmp_path / "auth.db"))
    monkeypatch.setattr(main_module.settings, "auth_invite_code", "test-invite")
    monkeypatch.setattr(main_module.settings, "auth_admin_email", "admin@example.com")
    with TestClient(app) as client:
        config = client.get("/api/auth/config")
        assert config.status_code == 200
        assert config.json()["admin_email"] == "admin@example.com"
        denied = client.post(
            "/api/auth/register",
            json={
                "display_name": "Researcher",
                "email": "researcher@example.com",
                "password": "correct-horse-battery",
                "invite_code": "wrong",
            },
        )
        assert denied.status_code == 403
        registered = client.post(
            "/api/auth/register",
            json={
                "display_name": "Researcher",
                "email": "researcher@example.com",
                "password": "correct-horse-battery",
                "invite_code": "test-invite",
            },
        )
        assert registered.status_code == 201
        assert client.get("/api/auth/me").json()["email"] == "researcher@example.com"
        assert client.post("/api/auth/logout").status_code == 204
        assert client.get("/api/auth/me").status_code == 401


def test_knowledge_status(authenticated_client) -> None:
    response = authenticated_client.get("/api/knowledge")
    assert response.status_code == 200
    assert "ready" in response.json()
    assert "chunks" in response.json()


def test_chat_validates_empty_input(authenticated_client) -> None:
    response = authenticated_client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_uploads_a_validated_chat_attachment(
    tmp_path, monkeypatch, authenticated_client
) -> None:
    test_settings = main_module.settings.model_copy(
        update={
            "attachments_dir": tmp_path / "attachments",
            "vision_input_enabled": True,
        }
    )
    monkeypatch.setattr(main_module, "attachment_store", AttachmentStore(test_settings))

    response = authenticated_client.post(
        "/api/attachments",
        files={"file": ("cell.png", PNG_1X1, "image/png")},
    )

    assert response.status_code == 201
    assert response.json()["kind"] == "image"
    assert response.json()["name"] == "cell.png"


def test_local_knowledge_search_excludes_configured_files(
    tmp_path, monkeypatch, authenticated_client
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    uploads_dir = tmp_path / "uploads"
    knowledge_dir.mkdir()
    uploads_dir.mkdir()
    (knowledge_dir / "demo_knowledge.md").write_text(
        "EGFR demo content that must be excluded.", encoding="utf-8"
    )
    (knowledge_dir / "validated_egfr.md").write_text(
        "EGFR C797S can cause resistance to covalent inhibitors.", encoding="utf-8"
    )
    test_settings = main_module.settings.model_copy(
        update={
            "knowledge_dir": knowledge_dir,
            "uploads_dir": uploads_dir,
            "knowledge_exclude_files": "demo_knowledge.md",
            "embedding_provider": "local",
        }
    )
    monkeypatch.setattr(main_module, "knowledge_store", KnowledgeStore(test_settings))

    response = authenticated_client.get(
        "/api/knowledge/search", params={"query": "EGFR C797S", "top_k": 2}
    )
    assert response.status_code == 200
    assert response.json()
    assert response.json()[0]["source_type"] == "local_knowledge"
    assert {row["title"] for row in response.json()} == {"validated_egfr.md"}


def test_training_status_is_available_to_local_client(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "training_api_enabled", True)
    with TestClient(app) as client:
        response = client.get("/api/training/status")
    assert response.status_code == 200
    assert response.json()["status"] in {
        "idle",
        "queued",
        "running",
        "skipped_insufficient_data",
        "dry_run_validated",
        "trained_unvalidated",
        "failed",
    }
