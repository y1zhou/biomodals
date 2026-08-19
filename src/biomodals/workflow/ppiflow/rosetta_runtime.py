"""Rosetta task implementations for the PPIFlow workflow."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast
from uuid import UUID

import orjson
import polars as pl

from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    execute_rosetta_task,
)
from biomodals.execution import PullTaskClaim, WorkerAssignmentRecord
from biomodals.execution.pull_worker import drive_pull_worker, size_pull_worker_pool
from biomodals.helper.app_run import AppRunLayout, volume_app_output
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    ExecutionArtifact,
    InlineBytes,
)
from biomodals.workflow.ppiflow import manifests as ppiflow_manifests
from biomodals.workflow.ppiflow import staging as ppiflow_staging
from biomodals.workflow.ppiflow.runtime_context import (
    ROSETTA_OUTPUT_MOUNTPOINT,
    ROSETTA_OUTPUT_VOLUME_NAME,
    SOURCE_VOLUMES,
)
from biomodals.workflow.ppiflow.runtime_context import (
    SOURCE_VOLUME_ROOTS as PPI_FLOW_SOURCE_VOLUME_ROOTS,
)
from biomodals.workflow.ppiflow.runtime_context import (
    reload_source_volumes as _reload_ppiflow_source_volumes,
)
from biomodals.workflow.ppiflow.runtime_support import (
    candidate_manifest_frame_from_inputs as _candidate_manifest_frame_from_inputs,
)
from biomodals.workflow.ppiflow.runtime_support import (
    config_int as _config_int,
)
from biomodals.workflow.ppiflow.runtime_support import (
    file_sha256 as _file_sha256,
)
from biomodals.workflow.ppiflow.runtime_support import (
    optional_config_int as _optional_config_int,
)
from biomodals.workflow.ppiflow.runtime_support import (
    write_candidate_manifest_output as _write_candidate_manifest_output,
)

_ROSETTA_PLAN_SCHEMA_VERSION = 1
ROSETTA_OUTPUT_VOLUME = SOURCE_VOLUMES[ROSETTA_OUTPUT_VOLUME_NAME]
APP_RUN_OUTPUT_STRUCTURE_PATTERNS = (
    "*.pdb",
    "*.cif",
    "**/*.pdb",
    "**/*.cif",
    "outputs/*.pdb",
    "outputs/**/*.pdb",
    "outputs/*.cif",
    "outputs/**/*.cif",
)


def _resolve_rosetta_config_text(value: str, field_name: str) -> str:
    config_path = Path(value).expanduser()
    has_newline = "\n" in value
    looks_like_path = (
        config_path.suffix in {".xml", ".flags"} or "/" in value or "\\" in value
    )
    if looks_like_path and not has_newline and config_path.exists():
        return config_path.read_text(encoding="utf-8")
    if looks_like_path and not has_newline:
        raise FileNotFoundError(
            f"Rosetta {field_name} path was not found locally or in the mounted "
            f"container filesystem: {value}"
        )
    return value


def _rosetta_plan_artifact(plan: Mapping[str, object]) -> AppOutput:
    """Serialize one immutable PPIFlow Rosetta Task plan."""
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise TypeError("Rosetta plan Tasks must be a list")
    return AppOutput(
        name="rosetta_task_plan",
        kind=ArtifactKind.TABLE,
        storage=InlineBytes(
            data=orjson.dumps(
                plan,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            ),
            filename="rosetta_task_plan.json",
            media_type="application/json",
        ),
        metadata={
            "task_count": len(tasks),
            "run_name": str(plan["run_name"]),
            "run_id": str(plan["run_id"]),
        },
    )


def _load_rosetta_plan(path: Path) -> dict[str, object]:
    """Load and validate one materialized PPIFlow Rosetta Task plan."""
    value = orjson.loads(path.read_bytes())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != _ROSETTA_PLAN_SCHEMA_VERSION
    ):
        raise ValueError("PPIFlow Rosetta task plan schema is unsupported")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("PPIFlow Rosetta task plan has no Tasks")
    specs = tuple(RosettaTaskSpec.from_dict(task) for task in tasks)
    if len(specs) != value.get("num_jobs"):
        raise ValueError("PPIFlow Rosetta Task count does not match its plan")
    return cast(dict[str, object], value)


def _read_rosetta_plan_artifacts(
    artifacts: Sequence[ExecutionArtifact],
) -> dict[str, object]:
    """Load the single Rosetta plan from a materialized workflow artifact."""
    if len(artifacts) != 1:
        raise ValueError(f"Expected one Rosetta task plan, found {len(artifacts)}")
    return _load_rosetta_plan(
        ppiflow_staging.artifact_mount_path(
            artifacts[0],
            PPI_FLOW_SOURCE_VOLUME_ROOTS,
        )
    )


def _rosetta_worker_policy(
    num_jobs: int,
    config: Mapping[str, object],
) -> tuple[int, int, int]:
    """Derive Rosetta pull workers from the Run's total container ceiling."""
    if num_jobs < 1:
        raise ValueError("Rosetta requires at least one Task")
    worker_count, claim_capacity = size_pull_worker_pool(
        num_jobs,
        max_worker_calls=_config_int(config, "_max_containers", 16),
        max_parallel_per_worker=30,
    )
    return worker_count, claim_capacity, claim_capacity


def prepare_ppiflow_rosetta_stage(
    *,
    artifacts: list[ExecutionArtifact],
    candidate_manifests: list[ExecutionArtifact] | None = None,
    config: dict[str, object],
    step_name: str,
    run_name: str,
    run_id: str,
    node_id: str,
) -> AppRunResult:
    """Stage Rosetta inputs and publish a finite pull-worker Task plan."""
    _reload_ppiflow_source_volumes()
    selected = ppiflow_staging.select_structure_files_from_artifacts(
        artifacts=artifacts,
        volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
        patterns=None,
        max_files=_optional_config_int(config, "max_structures"),
    )
    candidate_structures = ppiflow_staging.candidate_structure_files_from_selected(
        selected,
        manifest_frame=_candidate_manifest_frame_from_inputs(
            candidate_manifests or [],
            selected,
            step_name="RosettaInput",
        ),
    )
    if not candidate_structures:
        raise ValueError(f"{step_name} requires at least one Rosetta input")
    safe_run_name = sanitize_filename(run_name)
    safe_run_id = sanitize_filename(f"{run_id}-{node_id}")
    layout = AppRunLayout.from_run_root(
        Path(ROSETTA_OUTPUT_MOUNTPOINT) / f"{safe_run_name}-{safe_run_id}"
    )
    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    rosetta_script = config.get("rosetta_script")
    if rosetta_script is not None and not isinstance(rosetta_script, str):
        raise TypeError("rosetta_script must be text")
    script_content = (
        None
        if not rosetta_script
        else _resolve_rosetta_config_text(rosetta_script, "rosetta_script")
    )
    remote_script = None
    if script_content is not None:
        remote_script = "inputs/_script/workflow.xml"
        script_path = layout.run_root / remote_script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content, encoding="utf-8")
    flags_file = config.get("flags_file")
    if flags_file is not None and not isinstance(flags_file, str):
        raise TypeError("flags_file must be text")
    flags_content = (
        None
        if not flags_file
        else _resolve_rosetta_config_text(flags_file, "flags_file")
    )
    remote_flags = None
    if flags_content is not None:
        remote_flags = "inputs/_flags/workflow.flags"
        flags_path = layout.run_root / remote_flags
        flags_path.parent.mkdir(parents=True, exist_ok=True)
        flags_path.write_text(flags_content, encoding="utf-8")

    rosetta_rows = ppiflow_staging.rosetta_job_manifest_rows(
        candidate_structures,
        rosetta_binary=str(config.get("rosetta_binary", "relax")),
        rosetta_script=remote_script,
        flags_file=remote_flags,
    )
    task_specs = []
    for row, structure in zip(rosetta_rows, candidate_structures, strict=True):
        remote_pdb = str(row["pdb"])
        pdb_path = layout.run_root / remote_pdb
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.write_bytes(structure.data)
        task_specs.append(
            RosettaTaskSpec(
                task_key=str(row["candidate_id"]),
                index=_config_int(row, "index", 0),
                binary=str(row["binary"]),
                pdb=remote_pdb,
                rosetta_script=remote_script,
                flags_file=remote_flags,
                output_dir=str(row["expected_output_dir"]),
                worker_log=str(row["worker_log"]),
                expected_files=(str(row["expected_score_file"]),),
                input_sha256=hashlib.sha256(structure.data).hexdigest(),
                script_sha256=(
                    None
                    if script_content is None
                    else hashlib.sha256(script_content.encode()).hexdigest()
                ),
                flags_sha256=(
                    None
                    if flags_content is None
                    else hashlib.sha256(flags_content.encode()).hexdigest()
                ),
                candidate_id=str(row["candidate_id"]),
            )
        )
    job_manifest = ppiflow_staging.write_rosetta_job_manifest(
        rosetta_rows,
        layout.run_root / "rosetta_job_manifest.csv",
    )
    plan: dict[str, object] = {
        "schema_version": _ROSETTA_PLAN_SCHEMA_VERSION,
        "run_name": safe_run_name,
        "run_id": safe_run_id,
        "run_root": str(layout.run_root),
        "job_manifest": str(job_manifest),
        "num_jobs": len(task_specs),
        "tasks": [task.to_dict() for task in task_specs],
    }
    ROSETTA_OUTPUT_VOLUME.commit()
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[_rosetta_plan_artifact(plan)],
        metrics={
            "staged_candidates": len(task_specs),
        },
    )


def _rosetta_task_receipt(
    task: RosettaTaskSpec,
    task_fingerprint: str,
) -> AppOutput:
    """Return a small workflow-owned receipt for one validated Rosetta Task."""
    return AppOutput(
        name="rosetta_task_receipt",
        kind=ArtifactKind.REPORT,
        storage=InlineBytes(
            data=orjson.dumps(
                {
                    "task_key": task.task_key,
                    "task_fingerprint": task_fingerprint,
                    "candidate_id": task.candidate_id,
                    "expected_files": list(task.expected_files),
                },
                option=orjson.OPT_SORT_KEYS,
            ),
            filename=f"{sanitize_filename(task.task_key)}.json",
            media_type="application/json",
        ),
        metadata={"candidate_id": task.candidate_id or task.task_key},
    )


def run_ppiflow_rosetta_worker(
    coordinator,
    provider_call_id: str,
    run_name: str,
    run_id: str,
    claim_capacity: int,
    max_parallel: int,
) -> dict[str, int]:
    """Pull Rosetta Tasks from the workflow coordinator until none remain."""
    from biomodals.helper.shell import run_command

    layout = AppRunLayout.from_run_root(
        Path(ROSETTA_OUTPUT_MOUNTPOINT) / f"{run_name}-{run_id}"
    )

    def claim(request_id: str, capacity: int):
        return coordinator.claim_tasks.remote(
            provider_call_id,
            request_id,
            capacity,
        )

    def execute(assignment: WorkerAssignmentRecord) -> AppRunResult:
        payload = assignment.execution_payload
        if not isinstance(payload, Mapping):
            raise TypeError("Rosetta worker Task payload must be an object")
        task = RosettaTaskSpec.from_dict(payload.get("task"))
        try:
            execute_rosetta_task(
                run_root=layout.run_root,
                task=task,
                task_fingerprint=assignment.task_fingerprint,
                run_command=run_command,
            )
        except Exception as error:  # noqa: BLE001
            return AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=[str(error) or type(error).__name__],
                metrics={"candidate_id": task.candidate_id or task.task_key},
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[_rosetta_task_receipt(task, assignment.task_fingerprint)],
            metrics={"candidate_id": task.candidate_id or task.task_key},
        )

    def complete_and_claim(
        completions: tuple[
            tuple[WorkerAssignmentRecord, str, AppRunResult],
            ...,
        ],
        request_id: str,
        capacity: int,
    ) -> PullTaskClaim:
        return coordinator.complete_tasks_and_claim.remote(
            provider_call_id,
            tuple(
                (assignment.task_key, completion_request_id, result)
                for assignment, completion_request_id, result in completions
            ),
            request_id,
            capacity,
        )

    summary = drive_pull_worker(
        provider_call_id=UUID(provider_call_id),
        claim_capacity=claim_capacity,
        claim=claim,
        execute=execute,
        complete_and_claim=complete_and_claim,
        checkpoint_batch=ROSETTA_OUTPUT_VOLUME.commit,
        max_parallel=max_parallel,
    )
    return asdict(summary)


def _rosetta_task_outcomes_artifact(
    results: Mapping[str, AppRunResult],
    errors: Mapping[str, str],
) -> AppOutput:
    """Serialize terminal pull-Task outcomes for the remote finalizer."""
    return AppOutput(
        name="rosetta_task_outcomes",
        kind=ArtifactKind.TABLE,
        storage=InlineBytes(
            data=orjson.dumps(
                {
                    "schema_version": _ROSETTA_PLAN_SCHEMA_VERSION,
                    "succeeded": sorted(results),
                    "errors": {
                        task_key: errors[task_key] for task_key in sorted(errors)
                    },
                },
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            ),
            filename="rosetta_task_outcomes.json",
            media_type="application/json",
        ),
    )


def _read_rosetta_task_outcomes(
    artifacts: Sequence[ExecutionArtifact],
) -> tuple[set[str], dict[str, str]]:
    """Load one materialized Rosetta pull-Task outcome summary."""
    if len(artifacts) != 1:
        raise ValueError(
            f"Expected one Rosetta Task outcome artifact, found {len(artifacts)}"
        )
    path = ppiflow_staging.artifact_mount_path(
        artifacts[0],
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )
    value = orjson.loads(path.read_bytes())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != _ROSETTA_PLAN_SCHEMA_VERSION
        or not isinstance(value.get("succeeded"), list)
        or not isinstance(value.get("errors"), dict)
    ):
        raise ValueError("PPIFlow Rosetta Task outcomes are invalid")
    succeeded = {str(task_key) for task_key in value["succeeded"]}
    errors = {
        str(task_key): str(message) for task_key, message in value["errors"].items()
    }
    if succeeded & errors.keys():
        raise ValueError("Rosetta Task outcomes contain conflicting statuses")
    return succeeded, errors


def finalize_ppiflow_rosetta_stage(
    *,
    plan_artifacts: list[ExecutionArtifact],
    outcome_artifacts: list[ExecutionArtifact],
    config: dict[str, object],
    step_name: str,
    run_id: str,
    node_id: str,
) -> AppRunResult:
    """Validate Rosetta publications and emit the existing stage artifacts."""
    _reload_ppiflow_source_volumes()
    plan = _read_rosetta_plan_artifacts(plan_artifacts)
    succeeded_tasks, task_errors = _read_rosetta_task_outcomes(outcome_artifacts)
    task_specs = tuple(
        RosettaTaskSpec.from_dict(task) for task in cast(list[object], plan["tasks"])
    )
    worker_count, _claim_capacity, _max_parallel = _rosetta_worker_policy(
        len(task_specs),
        config,
    )
    expected_task_keys = {task.task_key for task in task_specs}
    if succeeded_tasks | task_errors.keys() != expected_task_keys:
        raise ValueError("Rosetta Task outcomes do not match the staged plan")

    run_root = Path(str(plan["run_root"]))
    job_manifest = Path(str(plan["job_manifest"]))
    row_frame = pl.read_csv(job_manifest, infer_schema_length=0)
    rows = []
    successful_candidates = 0
    warnings = []
    for row in row_frame.iter_rows(named=True):
        candidate_id = str(row["candidate_id"])
        expected_score = run_root / str(row["expected_score_file"])
        log_path = run_root / str(row["worker_log"])
        success = candidate_id in succeeded_tasks and expected_score.is_file()
        status = AppRunStatus.SUCCEEDED if success else AppRunStatus.FAILED
        error = task_errors.get(candidate_id)
        if error is None and not success:
            error = (
                "Rosetta Task reported success without its expected score file"
                if candidate_id in succeeded_tasks
                else "Rosetta Task did not publish a result"
            )
        if success:
            successful_candidates += 1
        elif error is not None:
            warnings.append(f"{candidate_id}: {error}")
        candidate_files = []
        if success:
            output_dir = run_root / str(row["expected_output_dir"])
            for path in sorted(output_dir.rglob("*")):
                if not path.is_file():
                    continue
                candidate_files.append(
                    ppiflow_manifests.candidate_file_record(
                        role=(
                            "structure"
                            if path.suffix.lower() in {".pdb", ".cif"}
                            else "score"
                        ),
                        volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                        app_volume_path=path.relative_to(
                            ROSETTA_OUTPUT_MOUNTPOINT
                        ).as_posix(),
                        path=path.relative_to(output_dir).as_posix(),
                        size_bytes=path.stat().st_size,
                        content_sha256=_file_sha256(path),
                    )
                )
        if log_path.is_file():
            candidate_files.append(
                ppiflow_manifests.candidate_file_record(
                    role="worker_log",
                    volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                    app_volume_path=log_path.relative_to(
                        ROSETTA_OUTPUT_MOUNTPOINT
                    ).as_posix(),
                    path=log_path.relative_to(run_root).as_posix(),
                    size_bytes=log_path.stat().st_size,
                    content_sha256=_file_sha256(log_path),
                    expected=False,
                )
            )
        rows.append(
            ppiflow_manifests.candidate_manifest_row(
                candidate_id=candidate_id,
                stage_name=step_name,
                stage_role="rosetta",
                operation_mode=str(config.get("rosetta_binary", "relax")),
                candidate_status=status.value,
                source_path=str(row["pdb"]),
                derived_path=str(row["expected_output_dir"]),
                error=error,
                files=candidate_files,
                summary={
                    "index": row["index"],
                    "num_pods": worker_count,
                },
            )
        )

    rosetta_files = [
        ArtifactFile(
            path=path.relative_to(run_root).as_posix(),
            role=("structure" if path.suffix.lower() in {".pdb", ".cif"} else "result"),
            size_bytes=path.stat().st_size,
            content_sha256=_file_sha256(path),
        )
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
    ]
    manifest_output = _write_candidate_manifest_output(
        run_id=run_id,
        node_id=node_id,
        step_name=step_name,
        rows=rows,
    )
    result_status = (
        AppRunStatus.SUCCEEDED if successful_candidates else AppRunStatus.FAILED
    )
    return AppRunResult(
        status=result_status,
        outputs=[
            volume_app_output(
                name="rosetta_outputs",
                kind=ArtifactKind.STRUCTURES,
                remote_path=str(plan["run_root"]),
                mount_root=ROSETTA_OUTPUT_MOUNTPOINT,
                volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                metadata={
                    "step_name": step_name,
                    "run_name": str(plan["run_name"]),
                    "run_id": str(plan["run_id"]),
                    "num_jobs": _config_int(plan, "num_jobs", 0),
                    "num_pods": worker_count,
                    "structure_patterns": APP_RUN_OUTPUT_STRUCTURE_PATTERNS,
                },
                files=rosetta_files,
            ),
            volume_app_output(
                name="rosetta_job_manifest",
                kind=ArtifactKind.TABLE,
                remote_path=str(plan["job_manifest"]),
                mount_root=ROSETTA_OUTPUT_MOUNTPOINT,
                volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                media_type="text/csv",
                metadata={
                    "step_name": step_name,
                    "rows": _config_int(plan, "num_jobs", 0),
                },
                files=[
                    ArtifactFile(
                        path=job_manifest.name,
                        role="job_manifest",
                        media_type="text/csv",
                        size_bytes=job_manifest.stat().st_size,
                        content_sha256=_file_sha256(job_manifest),
                    )
                ],
            ),
            manifest_output,
        ],
        warnings=warnings,
        metrics={
            "successful_candidates": successful_candidates,
            "failed_candidates": len(rows) - successful_candidates,
        },
    )
