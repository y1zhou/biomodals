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
    monkeypatch.setenv("BIOMODALS_PUBLIC_URL", "https://biomodals.internal")
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
            "--admin",
        ],
    )

    assert result.exit_code == 0, result.output
    link = result.output.strip()
    assert link.startswith("https://biomodals.internal/set-password#token=")
    assert result.output.count(link) == 1
    user = store.get_user_by_email("scientist@example.com")
    assert user is not None
    assert user.display_name == "A Scientist"
    assert user.is_admin is True


def test_first_cli_user_requires_admin_flag(monkeypatch, tmp_path) -> None:
    """An empty service cannot be bootstrapped without an administrator."""
    store = _configure(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "api",
            "admin",
            "create-user",
            "ordinary@example.com",
            "--display-name",
            "Ordinary",
        ],
    )

    assert result.exit_code == 1
    assert "first User must be provisioned as an administrator" in result.output
    store.initialize()
    assert store.list_users() == []


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
            "--admin",
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
    bootstrap = runner.invoke(
        app,
        [
            "api",
            "admin",
            "create-user",
            "admin@example.com",
            "--display-name",
            "Admin",
            "--admin",
        ],
    )
    assert bootstrap.exit_code == 0, bootstrap.output
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


def test_cli_promotes_and_demotes_without_removing_last_admin(
    monkeypatch,
    tmp_path,
) -> None:
    """Bootstrap and role commands preserve one active administrator."""
    store = _configure(monkeypatch, tmp_path)
    first = runner.invoke(
        app,
        [
            "api",
            "admin",
            "create-user",
            "first@example.com",
            "--display-name",
            "First",
            "--admin",
        ],
    )
    second = runner.invoke(
        app,
        [
            "api",
            "admin",
            "create-user",
            "second@example.com",
            "--display-name",
            "Second",
        ],
    )
    assert first.exit_code == second.exit_code == 0

    refused = runner.invoke(
        app,
        ["api", "admin", "demote-user", "first@example.com"],
    )
    promoted = runner.invoke(
        app,
        ["api", "admin", "promote-user", "second@example.com"],
    )
    demoted = runner.invoke(
        app,
        ["api", "admin", "demote-user", "first@example.com"],
    )

    assert refused.exit_code == 1
    assert "last active administrator" in refused.output
    assert promoted.output == "Promoted second@example.com\n"
    assert demoted.output == "Demoted first@example.com\n"
    first_user = store.get_user_by_email("first@example.com")
    second_user = store.get_user_by_email("second@example.com")
    assert first_user is not None and first_user.is_admin is False
    assert second_user is not None and second_user.is_admin is True
