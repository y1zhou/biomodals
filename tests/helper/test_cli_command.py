"""Tests for pure Biomodals CLI command builders."""

# ruff: noqa: D103

from biomodals.helper.cli_command import (
    build_workflow_run_command,
    resolve_workflow_entrypoint,
    select_modal_deployment_version,
)


def test_build_workflow_run_command_does_not_duplicate_dry_run() -> None:
    assert build_workflow_run_command(
        workflow_module="biomodals.workflow.shortmd_workflow",
        entrypoint="submit_shortmd_workflow",
        modal_mode="run",
        detach=False,
        dry_run=True,
        flags=["--dry-run", "/inputs"],
        python_executable="python",
    )[-2:] == ("--dry-run", "/inputs")


def test_select_modal_deployment_version_defaults_to_latest_or_validates_pin() -> None:
    history = (
        '[{"version":"v7","time_deployed":"now"},'
        '{"version":"v4","time_deployed":"earlier"}]'
    )

    assert select_modal_deployment_version(history) == 7
    assert select_modal_deployment_version(history, requested_version=4) == 4
    try:
        select_modal_deployment_version(history, requested_version=5)
    except ValueError as error:
        assert "version 5 is not available" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected an unavailable deployment version to fail")


def test_select_modal_deployment_version_rejects_invalid_history() -> None:
    for history in ("not json", "[]", '[{"version":"latest"}]'):
        try:
            select_modal_deployment_version(history)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"Expected invalid history to fail: {history}")


def test_resolve_workflow_entrypoint_uses_explicit_or_single_entrypoint() -> None:
    assert (
        resolve_workflow_entrypoint(
            workflow_name="shortmd",
            explicit_entrypoint="submit_shortmd_workflow",
            local_entrypoints=(),
        )
        == "submit_shortmd_workflow"
    )
    assert (
        resolve_workflow_entrypoint(
            workflow_name="shortmd",
            explicit_entrypoint=None,
            local_entrypoints=("submit_shortmd_workflow",),
        )
        == "submit_shortmd_workflow"
    )


def test_resolve_workflow_entrypoint_reports_ambiguous_workflows() -> None:
    try:
        resolve_workflow_entrypoint(
            workflow_name="ambiguous",
            explicit_entrypoint=None,
            local_entrypoints=("first", "second"),
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ambiguous workflow to raise")

    assert "contains multiple local entrypoints" in message
    assert "ambiguous::first" in message
    assert "ambiguous::second" in message
