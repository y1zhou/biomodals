"""Tests for shared workflow schema contracts."""

# ruff: noqa: D103

import ast
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from biomodals.schema import (
    AppConfig,
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    StorageKind,
    VolumePath,
    WorkflowArtifact,
)


def _valid_app_config(**overrides: object) -> AppConfig:
    values: dict[str, Any] = {
        "name": "demo",
        "package_name": "demo-package",
        "version": "1.0.0",
    }
    values.update(overrides)
    return AppConfig(**values)


def test_app_config_is_exported_from_schema_and_app_compatibility_module() -> None:
    from biomodals.app.config import AppConfig as CompatAppConfig

    schema_config = _valid_app_config()
    compat_config = CompatAppConfig(
        name="demo",
        package_name="demo-package",
        version="1.0.0",
    )

    assert issubclass(CompatAppConfig, AppConfig)
    assert (
        compat_config.model_dump(exclude={"output_volume", "output_volume_name"})
        == schema_config.model_dump()
    )
    assert compat_config.model_volume_subdir == "/demo"
    assert compat_config.git_clone_dir == Path("/opt/demo")
    assert compat_config.cuda_version_numeric == "12.8.0"
    assert compat_config.default_env["UV_TORCH_BACKEND"] == "cu128"
    assert compat_config.output_volume_name == "demo-outputs"
    assert hasattr(compat_config, "mounts")


def test_app_config_validates_source_reproducibility_and_runtime_bounds() -> None:
    with pytest.raises(ValidationError, match="repo_url"):
        AppConfig(name="missing-source", version="1.0.0")

    with pytest.raises(ValidationError, match="repo_commit_hash"):
        AppConfig(name="missing-version", package_name="demo-package")

    with pytest.raises(ValidationError, match="CUDA version must start"):
        _valid_app_config(cuda_version="12.8")

    with pytest.raises(ValidationError, match="CUDA 12.x"):
        _valid_app_config(gpu="B200+", cuda_version="cu128")

    with pytest.raises(ValidationError, match="Timeout must be between"):
        _valid_app_config(timeout=0)

    with pytest.raises(ValidationError, match="Timeout must be between"):
        _valid_app_config(timeout=999_999)


def test_app_config_records_dependency_apps_without_modal_imports() -> None:
    assert _valid_app_config().depends_on_apps == ()
    assert _valid_app_config(depends_on_apps=["gromacs"]).depends_on_apps == (
        "gromacs",
    )


def test_schema_modules_do_not_import_modal_app_or_workflow_packages() -> None:
    schema_dir = Path(__file__).parents[2] / "src" / "biomodals" / "schema"
    banned_import_roots = {"modal"}
    banned_import_prefixes = ("biomodals.app", "biomodals.workflow")

    for path in sorted(schema_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)

        banned = [
            module
            for module in imported_modules
            if module.split(".", 1)[0] in banned_import_roots
            or module.startswith(banned_import_prefixes)
        ]
        assert banned == [], f"{path.name} imports runtime-only modules: {banned}"


def test_inline_bytes_round_trip() -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="packed",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    data=b"hello\n",
                    filename="report.txt",
                    media_type="text/plain",
                ),
            )
        ],
    )

    dumped = result.model_dump_json()
    loaded = AppRunResult.model_validate_json(dumped)

    assert loaded.outputs[0].storage.kind == StorageKind.INLINE_BYTES
    assert isinstance(loaded.outputs[0].storage, InlineBytes)
    assert loaded.outputs[0].storage.data == b"hello\n"
    assert loaded.outputs[0].storage.filename == "report.txt"
    assert "aGVsbG8K" not in dumped
    assert "archive_format" not in InlineBytes.model_fields


def test_inline_bytes_rejects_binary_data_and_archive_metadata() -> None:
    with pytest.raises(ValidationError, match="UTF-8"):
        InlineBytes(data=b"\xff\x00", filename="binary.bin")

    with pytest.raises(ValidationError, match="archive_format"):
        InlineBytes(data=b"text", filename="archive.zip", archive_format="zip")  # type: ignore[ty:unknown-argument]


def test_volume_path_rejects_absolute_and_traversal_paths() -> None:
    for unsafe_path in ("/absolute/out", "../out", "a/../out", r"a\b"):
        with pytest.raises(ValidationError, match="VolumePath.path"):
            VolumePath(volume_name="Workflow-outputs", path=unsafe_path)


def test_workflow_artifact_is_volume_backed() -> None:
    artifact = WorkflowArtifact(
        artifact_id="art-packed",
        producing_node_id="packed",
        kind=ArtifactKind.STRUCTURES,
        storage=VolumePath(
            volume_name="Workflow-outputs",
            path="ppiflow/run-1/artifacts/art-packed",
        ),
    )

    assert artifact.storage.path == "ppiflow/run-1/artifacts/art-packed"
