"""Tests for the API server CLI command."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from biomodals.cli import app

runner = CliRunner()


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
