"""Tests for AF3Score standalone app contracts."""

# ruff: noqa: D103

import inspect

from biomodals.app.score import af3score_app


def test_af3score_remote_functions_do_not_accept_path_payloads() -> None:
    for function_name in (
        "af3score_prepare",
        "af3score_run",
        "af3score_postprocess",
    ):
        signature = inspect.signature(getattr(af3score_app, function_name).get_raw_f())
        assert "paths" not in signature.parameters
        assert "Path" not in str(signature)
