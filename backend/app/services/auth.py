from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.schemas import AuthUser

PASSWORD_ITERATIONS = 240_000


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class DuplicateEmailError(ValueError):
    pass


class InvalidCredentialsError(ValueError):
    pass


class InvalidInviteError(ValueError):
    pass


class RegistrationDisabledError(ValueError):
    pass


class AuthStore:
    """SQLite-backed users and opaque browser sessions."""

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
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS owned_resources (
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(resource_type, resource_id)
                );
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_owned_resources_user
                    ON owned_resources(user_id, resource_type);
                """
            )

    def register(
        self,
        *,
        email: str,
        display_name: str,
        password: str,
        invite_code: str,
        expected_invite_code: str,
        registration_enabled: bool,
    ) -> AuthUser:
        if not registration_enabled:
            raise RegistrationDisabledError("当前未开放新账号注册。")
        if not expected_invite_code or not hmac.compare_digest(invite_code, expected_invite_code):
            raise InvalidInviteError("邀请码无效，请确认后重试。")
        timestamp = _timestamp(_now())
        user_id = secrets.token_hex(16)
        salt = secrets.token_bytes(16)
        password_hash = _hash_password(password, salt)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users(
                        id, email, display_name, password_salt, password_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, email, display_name, salt, password_hash, timestamp),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateEmailError("该邮箱已注册，请直接登录。") from exc
        return AuthUser(id=user_id, email=email, display_name=display_name, created_at=timestamp)

    def authenticate(self, *, email: str, password: str) -> AuthUser:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            _hash_password(password, b"\0" * 16)
            raise InvalidCredentialsError("邮箱或密码不正确。")
        actual = _hash_password(password, bytes(row["password_salt"]))
        if not hmac.compare_digest(actual, bytes(row["password_hash"])):
            raise InvalidCredentialsError("邮箱或密码不正确。")
        return self._user(row)

    def create_session(self, user_id: str, *, duration_days: int) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        expires_at = now + timedelta(days=max(1, min(duration_days, 365)))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions(token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (_hash_token(token), user_id, _timestamp(now), _timestamp(expires_at)),
            )
            connection.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (_timestamp(now),))
        return token

    def user_for_session(self, token: str) -> AuthUser | None:
        now = _timestamp(_now())
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT u.* FROM auth_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ?
                """,
                (_hash_token(token), now),
            ).fetchone()
        return self._user(row) if row is not None else None

    def revoke_session(self, token: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_hash_token(token),))

    def claim_resource(self, resource_type: str, resource_id: str, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO owned_resources(resource_type, resource_id, user_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (resource_type, resource_id, user_id, _timestamp(_now())),
            )

    def owns_resource(self, resource_type: str, resource_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM owned_resources
                WHERE resource_type = ? AND resource_id = ? AND user_id = ?
                """,
                (resource_type, resource_id, user_id),
            ).fetchone()
        return row is not None

    @staticmethod
    def _user(row: sqlite3.Row) -> AuthUser:
        return AuthUser(
            id=str(row["id"]),
            email=str(row["email"]),
            display_name=str(row["display_name"]),
            created_at=str(row["created_at"]),
        )


__all__ = [
    "AuthStore",
    "DuplicateEmailError",
    "InvalidCredentialsError",
    "InvalidInviteError",
    "RegistrationDisabledError",
]
