"""Opt-in first-administrator bootstrap without changing normal authentication."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest
from pwdlib import PasswordHash
from typer.testing import CliRunner

from biomodals.cli import app
from biomodals.service.api import create_deployed_app
from biomodals.service.auth import AuthService, hash_password
from biomodals.service.config import ServiceSettings
from biomodals.service.remote_execution import RemoteExecutionClient
from biomodals.service.store import ServiceStore, UserAlreadyExistsError

PASSWORD = "a unique bootstrap passphrase"  # noqa: S105


@pytest.fixture(scope="module")
def password_hash():
    """Use the same password policy and hashing command as an operator."""
    return hash_password(PASSWORD)


@pytest.fixture
def auth(tmp_path):
    """Keep all account state in an isolated database."""
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    return AuthService(store, frontend_url="http://localhost:5173")


def test_bootstrap_creates_enabled_admin_and_preserves_password(auth, password_hash):
    """The initial hash works with existing login and is not reset on restart."""
    assert auth.bootstrap_admin(
        " Admin@Example.com ", password_hash, active_job_limit=3
    )
    user = auth.store.get_user_by_email("admin@example.com")
    assert user is not None and user.active and user.is_admin
    assert user.password_hash == password_hash
    assert user.active_job_limit == 3
    session = auth.login("admin@example.com", PASSWORD)
    assert auth.authenticate(session.session_token).principal.is_admin
    assert not auth.bootstrap_admin("admin@example.com", "not a valid hash")
    assert auth.store.get_user_by_email("admin@example.com") == user
    assert auth.authenticate(session.session_token) is not None


@pytest.mark.parametrize("status", ["pending_setup", "disabled", "enabled"])
def test_any_existing_admin_bypasses_invalid_bootstrap(auth, password_hash, status):
    """Bootstrap never becomes an alternate way to recover or replace an admin."""
    if status == "enabled":
        auth.bootstrap_admin("existing@example.com", password_hash)
    else:
        auth.create_user("existing@example.com", display_name="Existing", is_admin=True)
        if status == "disabled":
            auth.disable_user("existing@example.com")
    before = auth.store.list_users()
    assert not auth.bootstrap_admin("not-an-email", "invalid hash")
    assert not auth.bootstrap_admin("replacement@example.com", "")
    assert auth.store.list_users() == before


def test_unconfigured_bootstrap_preserves_manual_provisioning(auth):
    """No bootstrap values means the existing password-link path is unchanged."""
    assert not auth.bootstrap_admin("", "")
    assert auth.store.list_users() == []
    link = auth.create_user("admin@example.com", display_name="Admin", is_admin=True)
    assert "/set-password#token=" in link.url


@pytest.mark.parametrize(
    ("email", "encoded_hash"),
    [
        ("admin@example.com", ""),
        ("", "hash-without-email"),
        ("invalid-email", "not-a-hash"),
        ("admin@example.com", "plaintext must not be accepted"),
        ("admin@example.com", "$argon2id$v=19$m=65536,t=3,p=4$invalid$invalid"),
    ],
)
def test_invalid_bootstrap_fails_without_creating_user(auth, email, encoded_hash):
    """Missing values and malformed hashes fail before leaving a locked account."""
    with pytest.raises(ValueError) as error:
        auth.bootstrap_admin(email, encoded_hash)
    if encoded_hash:
        assert encoded_hash not in str(error.value)
    assert auth.store.list_users() == []


def test_bootstrap_does_not_promote_email_collision(auth, password_hash):
    """An existing non-admin account must never be silently taken over."""
    auth.create_user("taken@example.com", display_name="Existing", is_admin=True)
    with auth.store._transaction() as connection:
        connection.execute("UPDATE users SET is_admin = 0")
    before = auth.store.list_users()
    with pytest.raises(UserAlreadyExistsError):
        auth.bootstrap_admin("taken@example.com", password_hash)
    assert auth.store.list_users() == before


def test_concurrent_bootstrap_inserts_only_one_admin(auth, password_hash):
    """Both callers can observe no admin; the transactional recheck picks one."""
    barrier = Barrier(2)

    def create(index):
        assert not auth.store.has_admin()
        barrier.wait(timeout=5)
        return auth.store.bootstrap_admin(
            email=f"admin{index}@example.com",
            password_hash=password_hash,
            active_job_limit=2,
            now=1_800_000_000,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(create, range(2))) == [False, True]
    assert len(auth.store.list_users()) == 1


def test_hash_cli_hides_password_and_needs_no_service_configuration(monkeypatch):
    """Hash generation neither opens a database nor needs Modal credentials."""
    monkeypatch.setenv("BIOMODALS_API_CONF_ENV", "/does-not-exist/bootstrap.env")
    result = CliRunner().invoke(
        app, ["api", "admin", "hash-password"], input=f"{PASSWORD}\n{PASSWORD}\n"
    )
    assert result.exit_code == 0, result.output
    assert PASSWORD not in result.output
    encoded = result.stdout.strip().splitlines()[-1]
    assert encoded.startswith("$argon2id$")
    assert PasswordHash.recommended().verify(PASSWORD, encoded)


@pytest.mark.parametrize("password", ["too short", "passwordpassword"])
def test_hash_cli_applies_existing_password_policy(password):
    """Bootstrap hash generation has the same password policy as password setup."""
    result = CliRunner().invoke(
        app, ["api", "admin", "hash-password"], input=f"{password}\n{password}\n"
    )
    assert result.exit_code == 1
    assert "$argon2id$" not in result.output
    assert password not in result.output


def test_deployed_startup_bootstrap_uses_private_config_and_normal_login(
    tmp_path, monkeypatch, password_hash
):
    """Exercise real app assembly and HTTP auth with only Modal preflight stubbed."""
    config = tmp_path / "service.env"
    config.write_text(
        "BIOMODALS_DEFAULT_ADMIN_EMAIL=file@example.com\n"
        f"BIOMODALS_DEFAULT_ADMIN_PASSWORD_HASH='{password_hash}'\n"
    )
    config.chmod(0o600)
    settings = ServiceSettings.from_environment({
        "BIOMODALS_API_CONF_ENV": str(config),
        "BIOMODALS_STATE_DIR": str(tmp_path / "state"),
        "BIOMODALS_CACHE_DIR": str(tmp_path / "cache"),
        "BIOMODALS_DEFAULT_ADMIN_EMAIL": "process@example.com",
        "MODAL_TOKEN_ID": "offline-test-id",
        "MODAL_TOKEN_SECRET": "offline-test-secret",
    })
    monkeypatch.setattr(
        ServiceSettings, "from_environment", classmethod(lambda cls: settings)
    )
    # Keep install_modal_credentials from leaking test credentials to later tests.
    monkeypatch.setenv("MODAL_TOKEN_ID", "offline-test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "offline-test-secret")

    async def preflight(self, deployment):
        pass

    monkeypatch.setattr(RemoteExecutionClient, "preflight", preflight)

    async def run():
        deployed_app = create_deployed_app()
        async with (
            deployed_app.router.lifespan_context(deployed_app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=deployed_app),
                base_url="http://localhost:5173",
            ) as client,
        ):
            assert (await client.get("/api/v1/ready")).status_code == 200
            response = await client.post(
                "/api/v1/auth/login",
                json={"email": "process@example.com", "password": PASSWORD},
                headers={"Origin": "http://localhost:5173"},
            )
            assert response.status_code == 200, response.text
            assert (await client.get("/api/v1/auth/me")).json()["is_admin"] is True
            assert password_hash not in response.text

    asyncio.run(run())
