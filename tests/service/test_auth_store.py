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
    IssuedPasswordLink,
    PasswordPolicyError,
)
from biomodals.service.store import ServiceStore


class Clock:
    def __init__(self, now: int = 1_800_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def reset_token(link: IssuedPasswordLink) -> str:
    fragment = parse_qs(urlparse(link.urls[0]).fragment)
    return fragment["token"][0]


def make_auth(tmp_path: Path, clock: Clock) -> tuple[AuthService, ServiceStore]:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    return (
        AuthService(
            store,
            frontend_urls=("https://biomodals.internal",),
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
    link = auth.create_user(email, display_name="Alice", is_admin=True)
    auth.set_password(reset_token(link), password)


def test_setup_link_is_one_time_and_passwords_use_argon2id(tmp_path: Path) -> None:
    clock = Clock()
    auth, store = make_auth(tmp_path, clock)
    link = auth.create_user(
        "Alice@Example.com",
        display_name="Alice",
        is_admin=True,
    )
    token = reset_token(link)

    principal = auth.set_password(token, "correct horse battery staple").principal

    assert principal.email == "alice@example.com"
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    user = store.get_user_by_email(principal.email)
    assert user is not None
    assert user.password_hash is not None
    assert user.password_hash.startswith("$argon2id$")
    with pytest.raises(InvalidPasswordTokenError):
        auth.set_password(token, "another correct horse staple")


def test_alternative_password_urls_share_one_token_and_expiry(tmp_path: Path) -> None:
    clock = Clock()
    _auth, store = make_auth(tmp_path, clock)
    origins = ("https://biomodals.internal", "http://10.10.110.101")
    auth = AuthService(store, frontend_urls=origins, now=clock)
    setup = auth.create_user("alice@example.com", display_name="Alice", is_admin=True)
    assert setup.urls == tuple(
        f"{origin}/set-password#token={reset_token(setup)}" for origin in origins
    )
    assert setup.expires_at == clock.now + 3600
    # Redeem the HTTP alternative, then the HTTPS URL must also be spent.
    http_token = parse_qs(urlparse(setup.urls[1]).fragment)["token"][0]
    auth.set_password(http_token, "correct horse battery staple")
    with pytest.raises(InvalidPasswordTokenError):
        auth.set_password(reset_token(setup), "another correct horse staple")

    reset = auth.create_password_reset("alice@example.com")
    replacement = auth.create_password_reset("alice@example.com")
    assert replacement.urls == tuple(
        f"{origin}/set-password#token={reset_token(replacement)}" for origin in origins
    )
    for url in reset.urls:
        with pytest.raises(InvalidPasswordTokenError):
            auth.set_password(
                parse_qs(urlparse(url).fragment)["token"][0],
                "another correct horse staple",
            )
    auth.set_password(reset_token(replacement), "another correct horse staple")
    assert auth.login("alice@example.com", "another correct horse staple").principal


def test_create_user_rejects_an_oversized_display_name(tmp_path: Path) -> None:
    auth, _store = make_auth(tmp_path, Clock())

    with pytest.raises(ValueError, match="at most 120 characters"):
        auth.create_user(
            "alice@example.com",
            display_name="a" * 121,
        )


def test_password_policy_prefers_long_passphrases(tmp_path: Path) -> None:
    auth, _store = make_auth(tmp_path, Clock())
    token = reset_token(
        auth.create_user(
            "alice@example.com",
            display_name="Alice",
            is_admin=True,
        )
    )

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
    paths = (store.path, Path(f"{store.path}-wal"), Path(f"{store.path}-shm"))
    database_bytes = b"".join(path.read_bytes() for path in paths if path.exists())
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


def test_session_activity_is_persisted_at_a_bounded_frequency(tmp_path: Path) -> None:
    clock = Clock()
    auth, _store = make_auth(tmp_path, clock)
    activated_user(auth)
    issued = auth.login("alice@example.com", "correct horse battery staple")

    clock.now += 60
    first = auth.authenticate(issued.session_token)
    assert first is not None
    assert first.last_seen_at == 1_800_000_000

    clock.now += 5 * 60
    second = auth.authenticate(issued.session_token)
    assert second is not None
    assert second.last_seen_at == clock.now


def test_reset_and_disable_revoke_all_sessions(tmp_path: Path) -> None:
    clock = Clock()
    auth, _store = make_auth(tmp_path, clock)
    activated_user(auth)
    backup_link = auth.create_user(
        "backup-admin@example.com",
        display_name="Backup Admin",
        is_admin=True,
    )
    auth.set_password(
        reset_token(backup_link),
        "backup correct horse passphrase",
    )
    first = auth.login("alice@example.com", "correct horse battery staple")
    second = auth.login("alice@example.com", "correct horse battery staple")

    link = auth.create_password_reset("alice@example.com")
    auth.set_password(reset_token(link), "new correct horse passphrase")

    assert auth.authenticate(first.session_token) is None
    assert auth.authenticate(second.session_token) is None
    replacement = auth.login("alice@example.com", "new correct horse passphrase")
    auth.disable_user("alice@example.com")
    assert auth.authenticate(replacement.session_token) is None
