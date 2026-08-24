import pytest

from app.services.auth import (
    AuthStore,
    DuplicateEmailError,
    InvalidCredentialsError,
    InvalidInviteError,
)


def test_auth_store_enforces_invite_password_and_resource_ownership(tmp_path) -> None:
    store = AuthStore(tmp_path / "auth.db")
    store.initialize()

    with pytest.raises(InvalidInviteError):
        store.register(
            email="researcher@example.com",
            display_name="Researcher",
            password="correct-horse-battery",
            invite_code="wrong",
            expected_invite_code="invite-1",
            registration_enabled=True,
        )

    user = store.register(
        email="researcher@example.com",
        display_name="Researcher",
        password="correct-horse-battery",
        invite_code="invite-1",
        expected_invite_code="invite-1",
        registration_enabled=True,
    )
    with pytest.raises(DuplicateEmailError):
        store.register(
            email="researcher@example.com",
            display_name="Other",
            password="another-password",
            invite_code="invite-1",
            expected_invite_code="invite-1",
            registration_enabled=True,
        )
    with pytest.raises(InvalidCredentialsError):
        store.authenticate(email=user.email, password="wrong-password")

    assert store.authenticate(email=user.email, password="correct-horse-battery").id == user.id
    token = store.create_session(user.id, duration_days=30)
    assert store.user_for_session(token).id == user.id
    store.claim_resource("attachment", "attachment-1", user.id)
    assert store.owns_resource("attachment", "attachment-1", user.id) is True
    assert store.owns_resource("attachment", "attachment-1", "someone-else") is False
    store.revoke_session(token)
    assert store.user_for_session(token) is None
