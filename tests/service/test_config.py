"""Local service configuration contracts."""

# ruff: noqa: D103

import pytest

from biomodals.service.config import ServiceSettings

ENVIRONMENT_KEYS = (
    "BIOMODALS_STATE_DIR",
    "BIOMODALS_CACHE_DIR",
    "BIOMODALS_CACHE_MAX_BYTES",
    "BIOMODALS_FRONTEND_URL",
    "BIOMODALS_ALLOWED_ORIGIN",
    "BIOMODALS_SECURE_COOKIES",
    "BIOMODALS_MODAL_ENVIRONMENT",
    "BIOMODALS_GROMACS_APP",
    "BIOMODALS_GROMACS_ACTIVE_LIMIT",
    "BIOMODALS_RECONCILE_SECONDS",
    "BIOMODALS_INTERMEDIATE_RETENTION_DAYS",
)


def test_local_defaults_are_safe_and_cleanup_is_disabled(monkeypatch) -> None:
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = ServiceSettings.from_environment()

    assert settings.database_path.as_posix() == ".biomodals/state/service.sqlite3"
    assert settings.cache_dir.as_posix() == ".biomodals/cache"
    assert settings.frontend_url == "http://localhost:5173"
    assert settings.allowed_origin == settings.frontend_url
    assert settings.secure_cookies is False
    assert settings.modal_environment == "main"
    assert settings.intermediate_retention_days is None


def test_host_and_modal_settings_are_explicitly_configurable(monkeypatch) -> None:
    monkeypatch.setenv("BIOMODALS_STATE_DIR", "/srv/biomodals/state")
    monkeypatch.setenv("BIOMODALS_CACHE_DIR", "/srv/biomodals/cache")
    monkeypatch.setenv("BIOMODALS_SECURE_COOKIES", "true")
    monkeypatch.setenv("BIOMODALS_MODAL_ENVIRONMENT", "department")
    monkeypatch.setenv("BIOMODALS_INTERMEDIATE_RETENTION_DAYS", "14")

    settings = ServiceSettings.from_environment()

    assert settings.database_path.as_posix() == "/srv/biomodals/state/service.sqlite3"
    assert settings.cache_dir.as_posix() == "/srv/biomodals/cache"
    assert settings.secure_cookies is True
    assert settings.modal_environment == "department"
    assert settings.intermediate_retention_days == 14


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("BIOMODALS_CACHE_MAX_BYTES", "0"),
        ("BIOMODALS_RECONCILE_SECONDS", "0"),
        ("BIOMODALS_INTERMEDIATE_RETENTION_DAYS", "-1"),
    ],
)
def test_positive_settings_fail_closed(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        ServiceSettings.from_environment()
