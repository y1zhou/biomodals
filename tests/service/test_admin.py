"""Tests for manual API user administration."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from biomodals.cli import app
from biomodals.service.auth import AuthService, InvalidPasswordTokenError
from biomodals.service.store import ServiceStore

runner = CliRunner()


def _configure(monkeypatch, tmp_path) -> ServiceStore:
    monkeypatch.setenv("BIOMODALS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("BIOMODALS_FRONTEND_URL", "https://biomodals.internal")
    return ServiceStore(tmp_path / "state" / "service.sqlite3")


def _token(link: str) -> str:
    return parse_qs(urlsplit(link).fragment)["token"][0]


def test_create_user_prints_setup_link_once(monkeypatch, tmp_path) -> None:
    """Creation prints one link and stores normalized user metadata."""
    store = _configure(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "api",
            "admin",
            "create-user",
            "Scientist@Example.com",
            "--display-name",
            "A Scientist",
        ],
    )

    assert result.exit_code == 0, result.output
    link = result.output.strip()
    assert link.startswith("https://biomodals.internal/set-password#token=")
    assert result.output.count(link) == 1
    user = store.get_user_by_email("scientist@example.com")
    assert user is not None
    assert user.display_name == "A Scientist"


def test_reset_password_replaces_prior_link(monkeypatch, tmp_path) -> None:
    """Reset replaces the setup token and prints only the new link."""
    store = _configure(monkeypatch, tmp_path)
    created = runner.invoke(
        app,
        [
            "api",
            "admin",
            "create-user",
            "scientist@example.com",
            "--display-name",
            "Scientist",
        ],
    )
    first_link = created.output.strip()

    reset = runner.invoke(
        app,
        ["api", "admin", "reset-password", "scientist@example.com"],
    )

    assert reset.exit_code == 0, reset.output
    second_link = reset.output.strip()
    assert second_link != first_link
    assert reset.output.count(second_link) == 1
    auth = AuthService(store, frontend_url="https://biomodals.internal")
    with pytest.raises(InvalidPasswordTokenError):
        auth.set_password(_token(first_link), "a long unique passphrase")
    assert auth.set_password(_token(second_link), "a long unique passphrase")


def test_disable_user_revokes_access(monkeypatch, tmp_path) -> None:
    """Disabling a user immediately revokes their active browser session."""
    store = _configure(monkeypatch, tmp_path)
    created = runner.invoke(
        app,
        [
            "api",
            "admin",
            "create-user",
            "scientist@example.com",
            "--display-name",
            "Scientist",
        ],
    )
    auth = AuthService(store, frontend_url="https://biomodals.internal")
    auth.set_password(_token(created.output.strip()), "a long unique passphrase")
    session = auth.login("scientist@example.com", "a long unique passphrase")

    result = runner.invoke(
        app,
        ["api", "admin", "disable-user", "scientist@example.com"],
    )

    assert result.exit_code == 0, result.output
    assert result.output == "Disabled scientist@example.com\n"
    assert auth.authenticate(session.session_token) is None


def test_unknown_user_reports_clean_error(monkeypatch, tmp_path) -> None:
    """Expected administration errors are concise and omit tracebacks."""
    _configure(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["api", "admin", "reset-password", "missing@example.com"],
    )

    assert result.exit_code == 1
    assert result.output == "Error: Active user not found\n"
