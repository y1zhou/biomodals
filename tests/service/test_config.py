"""Local service configuration contracts."""

# ruff: noqa: D103,S105

from pathlib import Path

import pytest

from biomodals.service.config import ServiceSettings

ENVIRONMENT_KEYS = (
    "BIOMODALS_STATE_DIR",
    "BIOMODALS_CACHE_DIR",
    "BIOMODALS_CACHE_WARNING_BYTES",
    "BIOMODALS_API_CONF_ENV",
    "BIOMODALS_PUBLIC_URL",
    "BIOMODALS_SECURE_COOKIES",
    "BIOMODALS_MODAL_ENVIRONMENT",
    "BIOMODALS_GROMACS_APP",
    "BIOMODALS_GROMACS_APP_VERSION",
    "BIOMODALS_GROMACS_ACTIVE_LIMIT",
    "BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT",
    "BIOMODALS_DEFAULT_USER_ACTIVE_JOB_LIMIT",
    "BIOMODALS_RECONCILE_SECONDS",
    "BIOMODALS_INTERMEDIATE_RETENTION_DAYS",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
)


def test_local_defaults_are_safe_and_cleanup_is_disabled(monkeypatch) -> None:
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = ServiceSettings.from_environment()

    assert settings.database_path.as_posix() == ".biomodals/state/service.sqlite3"
    assert settings.cache_dir.as_posix() == ".biomodals/cache"
    assert settings.public_url == "http://localhost:5173"
    assert settings.secure_cookies is False
    assert settings.modal_environment == "production"
    assert settings.intermediate_retention_days is None


def test_host_and_modal_settings_are_explicitly_configurable(monkeypatch) -> None:
    monkeypatch.setenv("BIOMODALS_STATE_DIR", "/srv/biomodals/state")
    monkeypatch.setenv("BIOMODALS_CACHE_DIR", "/srv/biomodals/cache")
    monkeypatch.setenv("BIOMODALS_SECURE_COOKIES", "true")
    monkeypatch.setenv("BIOMODALS_PUBLIC_URL", "https://biomodals.example")
    monkeypatch.setenv("BIOMODALS_MODAL_ENVIRONMENT", "department")
    monkeypatch.setenv("BIOMODALS_INTERMEDIATE_RETENTION_DAYS", "14")

    settings = ServiceSettings.from_environment()

    assert settings.database_path.as_posix() == "/srv/biomodals/state/service.sqlite3"
    assert settings.cache_dir.as_posix() == "/srv/biomodals/cache"
    assert settings.secure_cookies is True
    assert settings.modal_environment == "department"
    assert settings.intermediate_retention_days == 14


def test_explicit_env_file_is_loaded_and_process_environment_wins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "service.env"
    config_path.write_text(
        "\n".join((
            'BIOMODALS_PUBLIC_URL="https://from-file.example"',
            "BIOMODALS_SECURE_COOKIES=true",
            "BIOMODALS_MODAL_ENVIRONMENT=file-environment",
            "MODAL_TOKEN_ID=file-token-id",
            "MODAL_TOKEN_SECRET=file-token-secret",
        ))
    )
    config_path.chmod(0o600)
    monkeypatch.setenv("BIOMODALS_API_CONF_ENV", str(config_path))
    monkeypatch.setenv("BIOMODALS_MODAL_ENVIRONMENT", "process-environment")

    settings = ServiceSettings.from_environment()

    assert settings.public_url == "https://from-file.example"
    assert settings.modal_environment == "process-environment"
    assert settings.modal_token_id == "file-token-id"
    assert settings.modal_token_secret == "file-token-secret"
    assert settings.sources.has_process_override("BIOMODALS_MODAL_ENVIRONMENT")
    assert not settings.sources.has_process_override("BIOMODALS_PUBLIC_URL")


def test_configured_env_file_must_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BIOMODALS_API_CONF_ENV", str(tmp_path / "missing.env"))

    with pytest.raises(ValueError, match="does not exist"):
        ServiceSettings.from_environment()


def test_configured_env_file_must_be_private(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "service.env"
    config_path.write_text("BIOMODALS_PUBLIC_URL=http://localhost:5173\n")
    config_path.chmod(0o640)
    monkeypatch.setenv("BIOMODALS_API_CONF_ENV", str(config_path))

    with pytest.raises(ValueError, match="group or world"):
        ServiceSettings.from_environment()


def test_relative_paths_resolve_from_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "service.env"
    config_path.write_text("BIOMODALS_STATE_DIR=state\nBIOMODALS_CACHE_DIR=cache\n")
    config_path.chmod(0o600)
    monkeypatch.setenv("BIOMODALS_API_CONF_ENV", str(config_path))

    settings = ServiceSettings.from_environment()

    assert settings.state_dir == tmp_path / "state"
    assert settings.cache_dir == tmp_path / "cache"


@pytest.mark.parametrize(
    ("public_url", "secure_cookies"),
    [("https://example.test", "false"), ("http://localhost:5173", "true")],
)
def test_public_url_and_cookie_mode_must_agree(
    monkeypatch: pytest.MonkeyPatch,
    public_url: str,
    secure_cookies: str,
) -> None:
    monkeypatch.setenv("BIOMODALS_PUBLIC_URL", public_url)
    monkeypatch.setenv("BIOMODALS_SECURE_COOKIES", secure_cookies)

    with pytest.raises(ValueError, match="must agree"):
        ServiceSettings.from_environment()


def test_modal_credentials_are_required_for_backend_startup(monkeypatch) -> None:
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    settings = ServiceSettings.from_environment()

    with pytest.raises(ValueError, match="MODAL_TOKEN_ID.*MODAL_TOKEN_SECRET"):
        settings.require_modal_credentials()


def test_deployed_backend_refuses_to_start_without_modal_credentials(
    monkeypatch,
) -> None:
    from biomodals.service.api import create_deployed_app

    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="MODAL_TOKEN_ID.*MODAL_TOKEN_SECRET"):
        create_deployed_app()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("BIOMODALS_CACHE_WARNING_BYTES", "0"),
        ("BIOMODALS_RECONCILE_SECONDS", "0"),
        ("BIOMODALS_INTERMEDIATE_RETENTION_DAYS", "-1"),
    ],
)
def test_positive_settings_fail_closed(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        ServiceSettings.from_environment()


@pytest.mark.parametrize(
    "name",
    [
        "BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT",
        "BIOMODALS_DEFAULT_USER_ACTIVE_JOB_LIMIT",
    ],
)
def test_active_job_limits_accept_zero(monkeypatch, name: str) -> None:
    monkeypatch.setenv(name, "0")

    settings = ServiceSettings.from_environment()

    assert {
        "BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT": settings.global_active_job_limit,
        "BIOMODALS_DEFAULT_USER_ACTIVE_JOB_LIMIT": (
            settings.default_user_active_job_limit
        ),
    }[name] == 0
