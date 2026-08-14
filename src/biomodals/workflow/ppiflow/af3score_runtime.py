"""AF3Score task implementations for the PPIFlow workflow."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import orjson

from biomodals.app.fold.alphafold3.inference_inputs import (
    DECLARED_MODEL_IDENTITY,
)
from biomodals.app.score import af3score_app
from biomodals.app.score.af3score_execution import (
    af3score_staged_input_key,
    materialize_af3score_staged_inputs,
)
from biomodals.helper.app_run import AppRunLayout, volume_path_from_mount_path
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    WorkflowArtifact,
)
from biomodals.workflow.ppiflow import manifests, staging, tables
from biomodals.workflow.ppiflow.runtime_context import (
    AF3SCORE_OUTPUT_MOUNTPOINT,
    AF3SCORE_OUTPUT_VOLUME_NAME,
    SOURCE_VOLUME_ROOTS,
    SOURCE_VOLUMES,
    reload_source_volumes,
)
from biomodals.workflow.ppiflow.runtime_support import (
    candidate_manifest_frame_from_inputs,
    config_int,
    optional_config_int,
    patterns_from_config,
    write_candidate_manifest_output,
)


def prepare_ppiflow_af3score_stage(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None = None,
    config: dict[str, object],
    step_name: str,
    execution_run_name: str,
) -> AppRunResult:
    """Stage AF3Score candidates and publish its finite GPU Task plan."""
    staged, run_name, publication_key, staged_input_key = _stage_candidate_inputs(
        artifacts=artifacts,
        candidate_manifests=candidate_manifests,
        execution_run_name=execution_run_name,
        patterns=patterns_from_config(config, default=("*.pdb",)),
        max_files=optional_config_int(config, "max_structures"),
    )
    input_names = [str(record["input_name"]) for record in staged]
    if not input_names:
        raise ValueError(f"{step_name} requires at least one AF3Score input")
    input_digests = {
        Path(str(record["input_name"])).stem: str(
            cast(Mapping[str, object], record["scientific_payload"])["content_sha256"]
        )
        for record in staged
    }
    task_spec = af3score_app.af3score_prepare.get_raw_f()(
        run_name=run_name,
        staged_input_key=staged_input_key,
        input_files=input_names,
        input_digests=input_digests,
        publication_key=publication_key,
        num_jobs=config_int(config, "_max_gpu_containers", 16),
        prepare_workers=config_int(config, "prepare_workers", 8),
    )
    chunks_by_input: dict[str, dict[str, str]] = {}
    raw_chunks = _task_spec_value(task_spec, "chunk_specs")
    if not isinstance(raw_chunks, Sequence):
        raise TypeError("AF3Score chunk_specs must be a sequence")
    for raw_chunk in raw_chunks:
        chunk = _chunk_payload(raw_chunk)
        for path in sorted(Path(chunk["batch_pdb_dir"]).glob("*.pdb")):
            if path.name in chunks_by_input:
                raise ValueError(f"AF3Score input {path.name!r} appears in two batches")
            chunks_by_input[path.name] = chunk
    pending = _task_spec_value(task_spec, "pending")
    if not isinstance(pending, int):
        raise TypeError("AF3Score pending Task count must be an integer")
    if len(chunks_by_input) != pending:
        raise ValueError(
            "AF3Score prepared Task count does not match its batch directories: "
            f"{pending} pending, {len(chunks_by_input)} mapped"
        )
    candidates = [
        record | {"chunk": chunks_by_input.get(str(record["input_name"]))}
        for record in staged
    ]
    chunk_sizes: dict[str, int] = {}
    for candidate in candidates:
        chunk = candidate["chunk"]
        if isinstance(chunk, Mapping):
            batch_name = str(chunk["batch_name"])
            chunk_sizes[batch_name] = chunk_sizes.get(batch_name, 0) + 1
    for candidate in candidates:
        chunk = candidate["chunk"]
        if isinstance(chunk, dict):
            chunk["task_count"] = chunk_sizes[str(chunk["batch_name"])]
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            _plan_artifact({
                "candidates": candidates,
                "input_files": input_names,
                "input_digests": input_digests,
                "publication_key": publication_key,
                "staged_input_key": staged_input_key,
                "run_name": run_name,
            })
        ],
    )


def run_ppiflow_af3score_batch(
    *,
    run_name: str,
    batch_name: str,
    batch_json_dir: str,
    batch_pdb_dir: str,
    task_keys: list[str],
    input_names: list[str],
    input_digests: dict[str, str],
    publication_key: str,
) -> dict[str, dict[str, object]]:
    """Run one AF3Score GPU batch and report each owned scientific Task."""
    if len(task_keys) != len(input_names) or not task_keys:
        raise ValueError(
            "AF3Score batch Task keys and inputs must be nonempty and align"
        )
    af3score_app.af3score_run.get_raw_f()(
        run_name=run_name,
        batch_name=batch_name,
        batch_json_dir=batch_json_dir,
        batch_pdb_dir=batch_pdb_dir,
        input_digests=input_digests,
        publication_key=publication_key,
    )
    layout = AppRunLayout.from_run_root(
        Path(AF3SCORE_OUTPUT_MOUNTPOINT) / sanitize_filename(run_name)
    )
    results: dict[str, dict[str, object]] = {}
    for task_key, input_name in zip(task_keys, input_names, strict=True):
        input_id = Path(input_name).stem
        complete = af3score_app._input_publication_ready(
            layout.outputs_dir,
            input_id,
            publication_key=publication_key,
            input_sha256=input_digests[input_id],
        )
        result = (
            AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name=f"af3score_{sanitize_filename(task_key)}",
                        kind=ArtifactKind.SCORES,
                        storage=volume_path_from_mount_path(
                            str(layout.outputs_dir / input_id),
                            AF3SCORE_OUTPUT_MOUNTPOINT,
                            AF3SCORE_OUTPUT_VOLUME_NAME,
                        ),
                        metadata={
                            "candidate_id": task_key,
                            "input_name": input_name,
                            "run_name": run_name,
                        },
                    )
                ],
            )
            if complete
            else AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=[f"AF3Score output is incomplete for {task_key!r}"],
            )
        )
        results[task_key] = result.model_dump(mode="json")
    return results


def postprocess_ppiflow_af3score_stage(
    *,
    plan_artifacts: list[WorkflowArtifact],
    task_keys: list[str],
    step_name: str,
    run_id: str,
    node_id: str,
) -> dict[str, dict[str, object]]:
    """Postprocess a kernel-completed AF3Score Task collection."""
    plan = _read_plan_artifacts(plan_artifacts)
    run_name = str(plan["run_name"])
    input_files = plan["input_files"]
    candidates = plan["candidates"]
    input_digests = plan.get("input_digests")
    publication_key = plan.get("publication_key")
    staged_input_key = plan.get("staged_input_key")
    if (
        not isinstance(input_files, list)
        or not isinstance(candidates, list)
        or not isinstance(input_digests, dict)
        or not isinstance(publication_key, str)
        or not isinstance(staged_input_key, str)
    ):
        raise TypeError("AF3Score task plan contains invalid candidate data")
    candidate_by_id: dict[str, Mapping[str, object]] = {}
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            raise TypeError("AF3Score candidate plan entries must be objects")
        candidate_by_id[str(raw_candidate["candidate_id"])] = raw_candidate
    normalized_digests = {str(key): str(value) for key, value in input_digests.items()}
    if len(task_keys) != len(set(task_keys)) or not set(task_keys).issubset(
        candidate_by_id
    ):
        raise ValueError("AF3Score postprocess Task keys do not match its plan")
    SOURCE_VOLUMES[AF3SCORE_OUTPUT_VOLUME_NAME].reload()
    outputs_dir = AppRunLayout.from_run_root(
        Path(AF3SCORE_OUTPUT_MOUNTPOINT) / run_name
    ).outputs_dir
    completed_input_ids = [
        Path(str(candidate["input_name"])).stem
        for candidate in candidate_by_id.values()
        if af3score_app._input_summary_publication_ready(
            outputs_dir,
            Path(str(candidate["input_name"])).stem,
            publication_key=publication_key,
            input_sha256=normalized_digests[Path(str(candidate["input_name"])).stem],
        )
    ]
    metrics = af3score_app.af3score_postprocess.get_raw_f()(
        run_name=run_name,
        staged_input_key=staged_input_key,
        input_files=[str(value) for value in input_files],
        input_digests=normalized_digests,
        completed_input_ids=completed_input_ids,
        publication_key=publication_key,
    )
    metrics_csv = str(metrics["metrics_csv"])
    failed_input_ids = {str(value) for value in metrics["failed_input_ids"]}
    status = tables.score_table_status(
        requested_count=len(input_files),
        usable_rows=int(metrics.get("metrics_rows", 0)),
        failed_count=int(metrics.get("failed", 0)),
    )
    manifest_output = write_candidate_manifest_output(
        run_id=run_id,
        node_id=node_id,
        step_name=step_name,
        rows=[
            manifests.candidate_manifest_row(
                candidate_id=str(candidate["candidate_id"]),
                stage_name=step_name,
                stage_role="score",
                operation_mode="af3score",
                candidate_status=(
                    AppRunStatus.FAILED.value
                    if Path(str(candidate["input_name"])).stem in failed_input_ids
                    else AppRunStatus.SUCCEEDED.value
                ),
                source_path=str(candidate["input_name"]),
                derived_path=metrics_csv,
                files=(
                    []
                    if Path(str(candidate["input_name"])).stem in failed_input_ids
                    else [
                        manifests.candidate_file_record(
                            role="scores",
                            volume_name=AF3SCORE_OUTPUT_VOLUME_NAME,
                            app_volume_path=volume_path_from_mount_path(
                                metrics_csv,
                                AF3SCORE_OUTPUT_MOUNTPOINT,
                                AF3SCORE_OUTPUT_VOLUME_NAME,
                            ).path,
                        )
                    ]
                ),
                summary=metrics,
            )
            for candidate in candidate_by_id.values()
        ],
    )
    aggregate = AppRunResult(
        status=status,
        outputs=[
            AppOutput(
                name="af3score_metrics",
                kind=ArtifactKind.SCORES,
                storage=volume_path_from_mount_path(
                    metrics_csv,
                    AF3SCORE_OUTPUT_MOUNTPOINT,
                    AF3SCORE_OUTPUT_VOLUME_NAME,
                ),
                metadata={"step_name": step_name, "run_name": run_name} | dict(metrics),
            ),
            manifest_output,
        ],
    )
    successful = [
        task_key
        for task_key in task_keys
        if Path(str(candidate_by_id[task_key]["input_name"])).stem
        not in failed_input_ids
    ]
    first_success = successful[0] if successful else None
    return {
        task_key: AppRunResult(
            status=(
                AppRunStatus.SUCCEEDED
                if task_key in successful
                else AppRunStatus.FAILED
            ),
            outputs=(aggregate.outputs if task_key == first_success else []),
            warnings=(
                []
                if task_key in successful
                else [f"AF3Score output is incomplete for {task_key!r}"]
            ),
        ).model_dump(mode="json")
        for task_key in task_keys
    }


def _stage_candidate_inputs(
    *,
    artifacts: list[WorkflowArtifact],
    candidate_manifests: list[WorkflowArtifact] | None,
    execution_run_name: str,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> tuple[list[dict[str, object]], str, str, str]:
    reload_source_volumes()
    selected = staging.select_structure_files_from_artifacts(
        artifacts=artifacts,
        volume_roots=SOURCE_VOLUME_ROOTS,
        patterns=patterns,
        max_files=max_files,
    )
    candidates = staging.candidate_structure_files_from_selected(
        selected,
        manifest_frame=candidate_manifest_frame_from_inputs(
            candidate_manifests or [],
            selected,
            step_name="AF3ScoreInput",
        ),
    )
    planned = [
        (
            candidate,
            f"{sanitize_filename(candidate.candidate_id)}.pdb",
            hashlib.sha256(candidate.data).hexdigest(),
        )
        for candidate in candidates
    ]
    input_names = [pdb_name for _candidate, pdb_name, _digest in planned]
    if len(input_names) != len(set(input_names)):
        raise ValueError("Duplicate AF3Score staged input name")
    input_digests = {
        Path(pdb_name).stem: digest for _candidate, pdb_name, digest in planned
    }
    publication_key = hashlib.sha256(
        orjson.dumps(
            {
                "inputs": input_digests,
                "af3score": (
                    af3score_app.CONF.repo_commit_hash
                    or af3score_app.CONF.version
                    or "unknown"
                ),
                "model": DECLARED_MODEL_IDENTITY,
            },
            option=orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()
    run_name = f"{sanitize_filename(execution_run_name)}-{publication_key}"
    staged_input_key = af3score_staged_input_key(
        tuple((pdb_name, digest) for _candidate, pdb_name, digest in planned)
    )
    materialize_af3score_staged_inputs(
        AF3SCORE_OUTPUT_MOUNTPOINT,
        staged_input_key,
        tuple(
            (pdb_name, candidate.data, digest)
            for candidate, pdb_name, digest in planned
        ),
    )
    SOURCE_VOLUMES[AF3SCORE_OUTPUT_VOLUME_NAME].commit()
    return (
        [
            {
                "candidate_id": candidate.candidate_id,
                "input_name": pdb_name,
                "scientific_payload": {
                    "candidate_id": candidate.candidate_id,
                    "content_sha256": digest,
                    "source_path": candidate.source_path,
                },
            }
            for candidate, pdb_name, digest in planned
        ],
        run_name,
        publication_key,
        staged_input_key,
    )


def _task_spec_value(task_spec: object, name: str) -> object:
    if isinstance(task_spec, Mapping):
        return task_spec[name]
    return getattr(task_spec, name)


def _chunk_payload(chunk: object) -> dict[str, str]:
    if isinstance(chunk, Mapping):
        return {
            "batch_json_dir": str(chunk["batch_json_dir"]),
            "batch_name": str(chunk["batch_name"]),
            "batch_pdb_dir": str(chunk["batch_pdb_dir"]),
        }
    chunk_spec = cast(af3score_app.ChunkSpec, chunk)
    return {
        "batch_json_dir": str(chunk_spec.batch_json_dir),
        "batch_name": str(chunk_spec.batch_name),
        "batch_pdb_dir": str(chunk_spec.batch_pdb_dir),
    }


def _plan_artifact(plan: Mapping[str, object]) -> AppOutput:
    candidates = plan["candidates"]
    if not isinstance(candidates, Sequence):
        raise TypeError("AF3Score plan candidates must be a sequence")
    return AppOutput(
        name="af3score_task_plan",
        kind=ArtifactKind.TABLE,
        storage=InlineBytes(
            data=orjson.dumps(
                plan,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            ),
            filename="af3score_task_plan.json",
            media_type="application/json",
        ),
        metadata={
            "candidate_count": len(candidates),
            "run_name": str(plan["run_name"]),
        },
    )


def _read_plan_artifacts(
    artifacts: Sequence[WorkflowArtifact],
) -> dict[str, object]:
    if len(artifacts) != 1:
        raise ValueError(f"Expected one AF3Score task plan, found {len(artifacts)}")
    path = staging.artifact_mount_path(artifacts[0], SOURCE_VOLUME_ROOTS)
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("AF3Score task plan must be a JSON object")
    return value
