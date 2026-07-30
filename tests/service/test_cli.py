"""Tests for the API server CLI command."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from biomodals.cli import app
from biomodals.service.store import ServiceStore

runner = CliRunner()


def _legacy_service_database(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE users (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT,
                status TEXT NOT NULL,
                is_admin INTEGER NOT NULL,
                active_job_limit INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE password_tokens (
                token_digest BLOB PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                expires_at INTEGER NOT NULL
            );
            CREATE TABLE sessions (
                token_digest BLOB PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                csrf_digest BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                absolute_expires_at INTEGER NOT NULL
            );
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL REFERENCES users(user_id)
            );
            CREATE TABLE job_operations (
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                operation TEXT NOT NULL
            );
            CREATE TABLE service_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE workload_settings (
                workload TEXT PRIMARY KEY,
                modal_app_name TEXT,
                modal_app_version INTEGER,
                active_job_limit INTEGER,
                job_logs_visible_to_owner INTEGER
            );
            INSERT INTO users VALUES (
                '11111111-1111-4111-8111-111111111111',
                'admin@example.com',
                'Admin',
                'hash',
                'enabled',
                1,
                7,
                1,
                2
            );
            INSERT INTO password_tokens VALUES (
                X'01',
                '11111111-1111-4111-8111-111111111111',
                100
            );
            INSERT INTO sessions VALUES (
                X'02',
                '11111111-1111-4111-8111-111111111111',
                X'03',
                10,
                20,
                200
            );
            INSERT INTO jobs VALUES (
                '22222222-2222-4222-8222-222222222222',
                '11111111-1111-4111-8111-111111111111'
            );
            INSERT INTO job_operations VALUES (
                '22222222-2222-4222-8222-222222222222',
                'prepare_tpr_gpu'
            );
            INSERT INTO service_settings VALUES ('modal_environment', 'research');
            INSERT INTO workload_settings VALUES (
                'gromacs',
                'GromacsPinned',
                17,
                4,
                1
            );
            PRAGMA user_version = 3;
            """
        )


@pytest.mark.parametrize(
    ("arguments", "expected_host", "expected_port"),
    [
        ((), "127.0.0.1", 4144),
        (("--host", "192.0.2.10", "--port", "9000"), "192.0.2.10", 9000),
    ],
)
def test_api_serve_runs_factory_with_one_worker(
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
    expected_host: str,
    expected_port: int,
) -> None:
    """The CLI exposes network settings without exposing unsafe worker scaling."""
    call: dict[str, object] = {}

    def fake_run(application: str, **kwargs: object) -> None:
        call["application"] = application
        call.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    result = runner.invoke(app, ["api", "serve", *arguments])

    assert result.exit_code == 0, result.output
    assert call == {
        "application": "biomodals.service.api:create_deployed_app",
        "factory": True,
        "host": expected_host,
        "port": expected_port,
        "workers": 1,
    }


def test_execution_state_transition_requires_explicit_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The destructive transition never runs from an accidental invocation."""
    state_dir = tmp_path / "state"
    database_path = state_dir / "service.sqlite3"
    _legacy_service_database(database_path)
    monkeypatch.setenv("BIOMODALS_STATE_DIR", str(state_dir))

    result = runner.invoke(app, ["api", "transition-execution-state"])

    assert result.exit_code == 1
    assert "Re-run with '--yes'" in result.output
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_execution_state_transition_preserves_accounts_and_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only obsolete Job execution history is discarded during cutover."""
    state_dir = tmp_path / "state"
    database_path = state_dir / "service.sqlite3"
    _legacy_service_database(database_path)
    monkeypatch.setenv("BIOMODALS_STATE_DIR", str(state_dir))

    result = runner.invoke(
        app,
        ["api", "transition-execution-state", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "discarded 1 legacy Job(s)" in result.output
    store = ServiceStore(database_path)
    store.initialize()
    [user] = store.list_users()
    assert (user.email, user.display_name, user.is_admin, user.active_job_limit) == (
        "admin@example.com",
        "Admin",
        True,
        7,
    )
    assert store.get_service_setting("modal_environment") == "research"
    workload = store.get_workload_configuration("gromacs")
    assert workload is not None
    assert (
        workload.workload,
        workload.modal_app_name,
        workload.modal_app_version,
        workload.active_job_limit,
    ) == ("gromacs", "GromacsPinned", 17, 4)
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM password_tokens").fetchone()[0]
            == 1
        )
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert "job_operations" not in tables
        assert "execution_runs" in tables


def test_execution_state_transition_rejects_unknown_legacy_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The offline transition never guesses which unknown tables to preserve."""
    state_dir = tmp_path / "state"
    database_path = state_dir / "service.sqlite3"
    _legacy_service_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE unexpected_state (value TEXT)")
    monkeypatch.setenv("BIOMODALS_STATE_DIR", str(state_dir))

    result = runner.invoke(
        app,
        ["api", "transition-execution-state", "--yes"],
    )

    assert result.exit_code == 1
    assert "Legacy service database schema is unexpected" in result.output
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_execution_state_transition_rejects_an_unexpected_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The offline transition never guesses how to rewrite another schema."""
    state_dir = tmp_path / "state"
    database_path = state_dir / "service.sqlite3"
    ServiceStore(database_path).initialize()
    monkeypatch.setenv("BIOMODALS_STATE_DIR", str(state_dir))

    result = runner.invoke(
        app,
        ["api", "transition-execution-state", "--yes"],
    )

    assert result.exit_code == 1
    assert "Expected pre-release service database version 3, found 4" in result.output
