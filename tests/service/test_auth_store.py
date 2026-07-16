"""Authentication and session persistence contracts."""

# ruff: noqa: D101,D102,D103,D107,S107

import stat
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from biomodals.service.auth import (
    AuthService,
    InvalidCredentialsError,
    InvalidPasswordTokenError,
    PasswordPolicyError,
)
from biomodals.service.store import ServiceStore


class Clock:
    def __init__(self, now: int = 1_800_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def reset_token(link: str) -> str:
    fragment = parse_qs(urlparse(link).fragment)
    return fragment["token"][0]


def make_auth(tmp_path: Path, clock: Clock) -> tuple[AuthService, ServiceStore]:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    return (
        AuthService(
            store,
            frontend_url="https://biomodals.internal",
            now=clock,
        ),
        store,
    )


def activated_user(
    auth: AuthService,
    *,
    email: str = "alice@example.com",
    password: str = "correct horse battery staple",
) -> None:
    link = auth.create_user(email, display_name="Alice")
    auth.set_password(reset_token(link), password)


def test_setup_link_is_one_time_and_passwords_use_argon2id(tmp_path: Path) -> None:
    clock = Clock()
    auth, store = make_auth(tmp_path, clock)
    link = auth.create_user("Alice@Example.com", display_name="Alice")
    token = reset_token(link)

    principal = auth.set_password(token, "correct horse battery staple")

    assert principal.email == "alice@example.com"
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert store.get_user_by_email(principal.email).password_hash.startswith(
        "$argon2id$"
    )
    with pytest.raises(InvalidPasswordTokenError):
        auth.set_password(token, "another correct horse staple")


def test_password_policy_prefers_long_passphrases(tmp_path: Path) -> None:
    auth, _store = make_auth(tmp_path, Clock())
    token = reset_token(auth.create_user("alice@example.com", display_name="Alice"))

    with pytest.raises(PasswordPolicyError):
        auth.set_password(token, "Password1!")


def test_login_uses_generic_failures_and_stores_only_token_digests(
    tmp_path: Path,
) -> None:
    clock = Clock()
    auth, store = make_auth(tmp_path, clock)
    activated_user(auth)

    with pytest.raises(InvalidCredentialsError):
        auth.login("missing@example.com", "correct horse battery staple")
    with pytest.raises(InvalidCredentialsError):
        auth.login("not-an-email", "correct horse battery staple")
    with pytest.raises(InvalidCredentialsError):
        auth.login("alice@example.com", "totally incorrect passphrase")

    issued = auth.login("alice@example.com", "correct horse battery staple")
    session = auth.authenticate(issued.session_token)

    assert session is not None
    assert session.principal.email == "alice@example.com"
    assert auth.verify_csrf(session, issued.csrf_token)
    database_bytes = store.database_bytes_for_test()
    assert issued.session_token.encode() not in database_bytes
    assert issued.csrf_token.encode() not in database_bytes


def test_sessions_have_idle_and_absolute_expiry(tmp_path: Path) -> None:
    clock = Clock()
    auth, _store = make_auth(tmp_path, clock)
    activated_user(auth)

    idle_session = auth.login("alice@example.com", "correct horse battery staple")
    clock.now += 30 * 24 * 60 * 60 + 1
    assert auth.authenticate(idle_session.session_token) is None

    clock.now -= 1
    absolute_session = auth.login("alice@example.com", "correct horse battery staple")
    for _ in range(3):
        clock.now += 29 * 24 * 60 * 60
        assert auth.authenticate(absolute_session.session_token) is not None
    clock.now += 4 * 24 * 60 * 60
    assert auth.authenticate(absolute_session.session_token) is None


def test_reset_and_disable_revoke_all_sessions(tmp_path: Path) -> None:
    clock = Clock()
    auth, _store = make_auth(tmp_path, clock)
    activated_user(auth)
    first = auth.login("alice@example.com", "correct horse battery staple")
    second = auth.login("alice@example.com", "correct horse battery staple")

    link = auth.create_password_reset("alice@example.com")
    auth.set_password(reset_token(link), "new correct horse passphrase")

    assert auth.authenticate(first.session_token) is None
    assert auth.authenticate(second.session_token) is None
    replacement = auth.login("alice@example.com", "new correct horse passphrase")
    auth.disable_user("alice@example.com")
    assert auth.authenticate(replacement.session_token) is None
