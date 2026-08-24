import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.services.auth import AuthStore


@pytest.fixture
def authenticated_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "auth_store", AuthStore(tmp_path / "auth.db"))
    monkeypatch.setattr(main_module.settings, "auth_invite_code", "test-invite")
    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/auth/register",
            json={
                "display_name": "Test Researcher",
                "email": "researcher@example.com",
                "password": "correct-horse-battery",
                "invite_code": "test-invite",
            },
        )
        assert response.status_code == 201
        yield client
