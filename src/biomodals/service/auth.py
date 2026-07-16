"""Manual user provisioning and cookie-session authentication."""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from hmac import compare_digest
from uuid import UUID

from pwdlib import PasswordHash

from biomodals.service.store import (
    ServiceStore,
    StoredSession,
    UserNotFoundError,
    UserRecord,
)

PASSWORD_TOKEN_LIFETIME_SECONDS = 60 * 60
SESSION_IDLE_TIMEOUT_SECONDS = 30 * 24 * 60 * 60
SESSION_ABSOLUTE_LIFETIME_SECONDS = 90 * 24 * 60 * 60
MIN_PASSWORD_CHARACTERS = 15
MAX_PASSWORD_CHARACTERS = 128

# A fixed, valid Argon2id hash keeps unknown-user login work comparable without
# generating a new hash for every request. Its password is deliberately unused.
_DUMMY_PASSWORD_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$NWAzSCUcXbHXIS7/NijhdA$"  # noqa: S105
    "DAuQlxDQ8oy5+CsmeIpUVFkX1jjdtZ+lEazp2amD7yw"
)
_COMMON_PASSWORDS = frozenset({
    "adminadminadmin",
    "changemechangeme",
    "letmeinletmein",
    "passwordpassword",
    "qwertyqwertyqwerty",
    "thisisapassword",
    "welcome123456789",
})


class InvalidCredentialsError(ValueError):
    """Raised for every unsuccessful login without revealing the reason."""


class InvalidPasswordTokenError(ValueError):
    """Raised when a setup/reset token is invalid, expired, or already used."""


class PasswordPolicyError(ValueError):
    """Raised when a proposed password does not meet the local policy."""


@dataclass(frozen=True, slots=True)
class Principal:
    """Stable user identity exposed to API authorization code."""

    user_id: UUID
    email: str
    display_name: str


@dataclass(frozen=True, slots=True)
class IssuedSession:
    """Raw session and CSRF tokens returned exactly once at login."""

    session_token: str
    csrf_token: str
    principal: Principal


@dataclass(frozen=True, slots=True)
class AuthenticatedSession:
    """Authenticated request context with a digest for CSRF validation."""

    principal: Principal
    csrf_digest: bytes
    created_at: int
    last_seen_at: int
    absolute_expires_at: int


class AuthService:
    """Apply identity policy while delegating atomic state to SQLite."""

    def __init__(
        self,
        store: ServiceStore,
        *,
        frontend_url: str,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Configure persistence, frontend links, and an injectable clock."""
        self.store = store
        self.frontend_url = frontend_url.rstrip("/")
        self._now = now or (lambda: int(time.time()))
        self._password_hash = PasswordHash.recommended()

    def create_user(self, email: str, *, display_name: str) -> str:
        """Provision a user and return a one-hour password setup link."""
        normalized_email = _normalize_email(email)
        normalized_name = display_name.strip()
        if not normalized_name:
            raise ValueError("Display name is required")
        token = _new_token()
        now = self._now()
        self.store.create_user(
            email=normalized_email,
            display_name=normalized_name,
            token_digest=_token_digest(token),
            token_expires_at=now + PASSWORD_TOKEN_LIFETIME_SECONDS,
            now=now,
        )
        return self._password_link(token)

    def create_password_reset(self, email: str) -> str:
        """Replace prior reset links and return a one-hour password link."""
        user = self.store.get_user_by_email(_normalize_email(email))
        if user is None or not user.active:
            raise UserNotFoundError("Active user not found")
        token = _new_token()
        self.store.issue_password_token(
            user.user_id,
            token_digest=_token_digest(token),
            expires_at=self._now() + PASSWORD_TOKEN_LIFETIME_SECONDS,
        )
        return self._password_link(token)

    def set_password(self, token: str, password: str) -> Principal:
        """Consume a one-time link and revoke all existing browser sessions."""
        _validate_password(password)
        password_hash = self._password_hash.hash(password)
        user = self.store.set_password_from_token(
            _token_digest(token),
            password_hash=password_hash,
            now=self._now(),
        )
        if user is None:
            raise InvalidPasswordTokenError("Password link is invalid or expired")
        return _principal(user)

    def login(self, email: str, password: str) -> IssuedSession:
        """Verify a password and issue opaque server-side session credentials."""
        try:
            normalized_email = _normalize_email(email)
        except ValueError:
            normalized_email = ""
        user = self.store.get_user_by_email(normalized_email)
        stored_hash = (
            user.password_hash
            if user is not None and user.password_hash is not None
            else _DUMMY_PASSWORD_HASH
        )
        valid, replacement_hash = self._password_hash.verify_and_update(
            password,
            stored_hash,
        )
        if not valid or user is None or not user.active or user.password_hash is None:
            raise InvalidCredentialsError("Invalid email or password")

        session_token = _new_token()
        csrf_token = _new_token()
        now = self._now()
        created = self.store.create_session_if_password_matches(
            user.user_id,
            expected_password_hash=user.password_hash,
            replacement_password_hash=replacement_hash,
            token_digest=_token_digest(session_token),
            csrf_digest=_token_digest(csrf_token),
            now=now,
            absolute_expires_at=now + SESSION_ABSOLUTE_LIFETIME_SECONDS,
        )
        if not created:
            raise InvalidCredentialsError("Invalid email or password")
        return IssuedSession(
            session_token=session_token,
            csrf_token=csrf_token,
            principal=_principal(user),
        )

    def authenticate(self, session_token: str) -> AuthenticatedSession | None:
        """Authenticate and slide the idle deadline for one opaque token."""
        stored = self.store.authenticate_session(
            _token_digest(session_token),
            now=self._now(),
            idle_timeout_seconds=SESSION_IDLE_TIMEOUT_SECONDS,
        )
        return _authenticated_session(stored) if stored is not None else None

    def verify_csrf(self, session: AuthenticatedSession, csrf_token: str) -> bool:
        """Compare a request CSRF token with the digest bound to its session."""
        return compare_digest(session.csrf_digest, _token_digest(csrf_token))

    def logout(self, session_token: str) -> None:
        """Revoke one opaque browser session."""
        self.store.revoke_session(_token_digest(session_token))

    def disable_user(self, email: str) -> Principal:
        """Disable an account and revoke its sessions and password links."""
        user = self.store.disable_user(
            _normalize_email(email),
            now=self._now(),
        )
        return _principal(user)

    def _password_link(self, token: str) -> str:
        return f"{self.frontend_url}/reset-password#token={token}"


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


def _normalize_email(email: str) -> str:
    normalized = email.strip().casefold()
    local, separator, domain = normalized.partition("@")
    if not separator or not local or not domain or "@" in domain:
        raise ValueError("A valid email address is required")
    if any(character.isspace() for character in normalized):
        raise ValueError("A valid email address is required")
    return normalized


def _validate_password(password: str) -> None:
    if not MIN_PASSWORD_CHARACTERS <= len(password) <= MAX_PASSWORD_CHARACTERS:
        raise PasswordPolicyError(
            f"Password must contain {MIN_PASSWORD_CHARACTERS} to "
            f"{MAX_PASSWORD_CHARACTERS} characters"
        )
    if password.casefold() in _COMMON_PASSWORDS:
        raise PasswordPolicyError("Choose a less common password")


def _principal(user: UserRecord) -> Principal:
    return Principal(
        user_id=user.user_id,
        email=user.email,
        display_name=user.display_name,
    )


def _authenticated_session(stored: StoredSession) -> AuthenticatedSession:
    return AuthenticatedSession(
        principal=_principal(stored.user),
        csrf_digest=stored.csrf_digest,
        created_at=stored.created_at,
        last_seen_at=stored.last_seen_at,
        absolute_expires_at=stored.absolute_expires_at,
    )
