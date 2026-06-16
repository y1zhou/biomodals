"""Tests for pure Biomodals CLI command builders."""

# ruff: noqa: D103

from biomodals.helper.cli_command import (
    build_app_run_command,
    build_modal_deploy_command,
    build_workflow_run_command,
    modal_env_overrides,
)


def test_build_app_run_command_targets_path_or_entrypoint() -> None:
    assert build_app_run_command(
        app_path="src/biomodals/app/fold/demo_app.py",
        entrypoint="submit_demo_task",
        modal_mode="run",
        detach=True,
        flags=["--input", "demo.pdb"],
        python_executable="python",
    ) == (
        "python",
        "-m",
        "modal",
        "run",
        "-d",
        "src/biomodals/app/fold/demo_app.py::submit_demo_task",
        "--input",
        "demo.pdb",
    )


def test_build_workflow_run_command_forwards_dry_run_before_user_flags() -> None:
    assert build_workflow_run_command(
        workflow_module="biomodals.workflow.shortmd_workflow",
        entrypoint="submit_shortmd_workflow",
        modal_mode="run",
        detach=False,
        dry_run=True,
        flags=["/inputs", "--replicates", "1"],
        python_executable="python",
    ) == (
        "python",
        "-m",
        "modal",
        "run",
        "-m",
        "biomodals.workflow.shortmd_workflow::submit_shortmd_workflow",
        "--dry-run",
        "/inputs",
        "--replicates",
        "1",
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


def test_modal_env_overrides_only_contains_requested_values() -> None:
    assert modal_env_overrides(gpu="L40S", timeout=3600) == {
        "GPU": "L40S",
        "TIMEOUT": "3600",
    }
    assert modal_env_overrides(gpu=None, timeout=None) == {}


def test_build_modal_deploy_command() -> None:
    assert build_modal_deploy_command(
        app_path="src/biomodals/app/fold/demo_app.py",
        name="demo-prod",
        tag="v1",
    ) == (
        "modal",
        "deploy",
        "--name",
        "demo-prod",
        "--tag",
        "v1",
        "src/biomodals/app/fold/demo_app.py",
    )
