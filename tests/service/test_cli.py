"""Tests for the API server CLI command."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from biomodals.cli import app

runner = CliRunner()


def test_api_serve_runs_factory_with_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI exposes network settings without exposing unsafe worker scaling."""
    call: dict[str, object] = {}

    def fake_run(application: str, **kwargs: object) -> None:
        call["application"] = application
        call.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    result = runner.invoke(
        app,
        ["api", "serve", "--host", "192.0.2.10", "--port", "9000"],
    )

    assert result.exit_code == 0, result.output
    assert call == {
        "application": "biomodals.service.api:create_deployed_app",
        "factory": True,
        "host": "192.0.2.10",
        "port": 9000,
        "workers": 1,
    }
