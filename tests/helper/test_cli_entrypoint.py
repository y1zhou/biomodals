"""Tests for in-process deployed-entrypoint invocation."""

# ruff: noqa: D103

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from biomodals.helper.cli_entrypoint import (
    _parse_entrypoint_flags,
    invoke_local_entrypoint,
)


def test_parse_entrypoint_flags_accepts_required_options_and_scalar_values() -> None:
    def callback(
        input_json: str,
        out_dir: str | None = None,
        search_msa: bool = True,
        workers: int = 4,
        threshold: float = 0.5,
    ) -> None:
        pass

    assert _parse_entrypoint_flags(
        callback,
        [
            "--input-json",
            "input.json",
            "--out-dir",
            "outputs",
            "--no-search-msa",
            "--workers",
            "8",
            "--threshold",
            "0.25",
        ],
        hidden_parameters=frozenset(),
        program_name="test",
    ) == {
        "input_json": "input.json",
        "out_dir": "outputs",
        "search_msa": False,
        "workers": 8,
        "threshold": 0.25,
    }


def test_parse_entrypoint_flags_accepts_required_positionals() -> None:
    def callback(input_dir: str, replicates: int = 50) -> None:
        pass

    assert _parse_entrypoint_flags(
        callback,
        ["/inputs", "--replicates", "2"],
        hidden_parameters=frozenset(),
        program_name="test",
    ) == {
        "input_dir": "/inputs",
        "replicates": 2,
    }


def test_parse_entrypoint_flags_requires_mandatory_input() -> None:
    def callback(input_dir: str, replicates: int = 50) -> None:
        pass

    with pytest.raises(SystemExit) as error:
        _parse_entrypoint_flags(
            callback,
            [],
            hidden_parameters=frozenset(),
            program_name="test",
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], None),
        (["input.pdb"], "input.pdb"),
        (["--input-path", "input.pdb"], "input.pdb"),
    ],
)
def test_nullable_leading_input_accepts_positional_or_named_value(flags, expected):
    def callback(input_path: str | None = None, run_name: str | None = None) -> None:
        pass

    assert _parse_entrypoint_flags(
        callback,
        flags,
        hidden_parameters=frozenset(),
        program_name="test",
    ) == {"input_path": expected, "run_name": None}


@pytest.mark.parametrize(
    "flags",
    [["input.pdb", "extra.pdb"], ["--input-path", "input.pdb", "extra.pdb"]],
)
def test_nullable_leading_input_rejects_extra_or_duplicate_positionals(flags):
    def callback(input_path: str | None = None, run_name: str | None = None) -> None:
        pass

    with pytest.raises(SystemExit) as error:
        _parse_entrypoint_flags(
            callback,
            flags,
            hidden_parameters=frozenset(),
            program_name="test",
        )
    assert error.value.code == 2


def test_invoke_local_entrypoint_hides_and_injects_deployment_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("MODAL_ENVIRONMENT", "original")

    def callback(
        input_json: str,
        *,
        use_deployed_coordinator: bool = False,
        deployment_version: int = 1,
    ) -> None:
        calls.append({
            "input_json": input_json,
            "use_deployed_coordinator": use_deployed_coordinator,
            "deployment_version": deployment_version,
            "environment": os.environ.get("MODAL_ENVIRONMENT"),
        })

    module = SimpleNamespace(
        submit=SimpleNamespace(info=SimpleNamespace(raw_f=callback))
    )
    monkeypatch.setattr(
        "biomodals.helper.cli_entrypoint.importlib.import_module",
        lambda _module_name: module,
    )

    invoke_local_entrypoint(
        module_name="example.app",
        entrypoint_name="submit",
        flags=["--input-json", str(Path("input.json"))],
        overrides={
            "use_deployed_coordinator": True,
            "deployment_version": 7,
        },
        program_name="test",
        environment_name="production",
    )

    assert calls == [
        {
            "input_json": "input.json",
            "use_deployed_coordinator": True,
            "deployment_version": 7,
            "environment": "production",
        }
    ]
    assert os.environ["MODAL_ENVIRONMENT"] == "original"


def test_invoke_local_entrypoint_rejects_user_supplied_deployment_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def callback(input_json: str, deployment_version: int = 1) -> None:
        pass

    module = SimpleNamespace(
        submit=SimpleNamespace(info=SimpleNamespace(raw_f=callback))
    )
    monkeypatch.setattr(
        "biomodals.helper.cli_entrypoint.importlib.import_module",
        lambda _module_name: module,
    )

    with pytest.raises(SystemExit):
        invoke_local_entrypoint(
            module_name="example.app",
            entrypoint_name="submit",
            flags=[
                "--input-json",
                "input.json",
                "--deployment-version",
                "99",
            ],
            overrides={"deployment_version": 7},
            program_name="test",
        )
