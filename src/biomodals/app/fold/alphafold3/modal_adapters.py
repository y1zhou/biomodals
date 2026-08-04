"""Narrow Modal adapters retained by AlphaFold 3 workload code."""

# Protocol implementations inherit their method contracts.
# ruff: noqa: D102

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import modal

from biomodals.app.fold.alphafold3.inference_inputs import (
    PreparedInferenceRun,
    VolumeUpload,
)
from biomodals.app.fold.alphafold3.inference_pipeline import (
    InferenceBatchOutcome,
    InferenceExecutor,
)
from biomodals.app.fold.alphafold3.profile_builder import (
    plan_missing_profile_builds,
)
from biomodals.app.fold.alphafold3.profiles import profile_build_slot_budget
from biomodals.app.fold.alphafold3.seed_predictions import (
    ClaimedSeed,
    SeedClaimPlan,
    partition_claimed_seeds,
    seed_claim_plan_from_dict,
)


def execute_profile_setup(
    inspect_function: modal.Function,
    build_function: modal.Function,
    finalize_function: modal.Function,
    *,
    seqkit_threads: int,
    source_policy: str,
) -> dict[str, object]:
    """Build all missing profiles concurrently, then finalize their registry."""
    print("🧬 Inspecting fixed sharded database profiles...")
    inventory = inspect_function.remote()
    inputs = plan_missing_profile_builds(
        inventory,
        seqkit_threads,
        source_policy,
    )
    missing = [database_id for database_id, _, _ in inputs]
    budget = profile_build_slot_budget(len(inputs), seqkit_threads)
    print(
        "🧬 Effective profile-build fanout: "
        f"{budget['builder_containers']} containers, "
        f"{budget['maximum_effective_worker_slots']} configured worker slots."
    )

    results: list[dict[str, object] | BaseException] = []
    if missing:
        print(
            "🧬 Submitting missing database profiles concurrently: "
            f"{', '.join(missing)}"
        )
        results = list(
            build_function.starmap(
                inputs,
                return_exceptions=True,
            )
        )
        failures = [
            {
                "database_id": database_id,
                "error_type": type(result).__name__,
                "message": str(result),
            }
            for database_id, result in zip(missing, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise RuntimeError(
                "One or more sharded database builders failed after all "
                f"submitted builders completed: {failures}"
            )
    else:
        print("🧬 All fixed profiles are already valid; no builders submitted.")

    print("🧬 Running final inventory and workspace cleanup...")
    return {
        "status": "complete",
        "initial_inventory": inventory,
        "builder_results": results,
        "final_inventory": finalize_function.remote(),
    }


def _volume_file_state(
    volume: modal.Volume,
    path: str,
    expected: bytes,
) -> Literal["missing", "match", "conflict"]:
    offset = 0
    try:
        for chunk in volume.read_file(path):
            if not isinstance(chunk, bytes):
                raise TypeError(f"Volume returned non-bytes for {path}")
            end = offset + len(chunk)
            if end > len(expected) or expected[offset:end] != chunk:
                return "conflict"
            offset = end
    except FileNotFoundError:
        return "missing"
    return "match" if offset == len(expected) else "conflict"


def stage_inference_run(
    output_volume: modal.Volume,
    prepared: PreparedInferenceRun,
) -> None:
    """Publish immutable payloads once, with the staged-input marker last."""
    marker_path = prepared.staged_input.relative_path.as_posix()
    marker_state = _volume_file_state(
        output_volume,
        marker_path,
        prepared.staged_input.content,
    )
    if marker_state == "conflict":
        raise RuntimeError(
            "Existing staged-input marker conflicts with the prepared request: "
            f"{marker_path}"
        )
    if marker_state == "match":
        pending_uploads = []
        for upload in prepared.payload_uploads:
            path = upload.relative_path.as_posix()
            if _volume_file_state(output_volume, path, upload.content) != "match":
                pending_uploads.append(upload)
        if not pending_uploads:
            return
    else:
        pending_uploads = list(prepared.payload_uploads)

    with output_volume.batch_upload(force=True) as batch:
        for upload in pending_uploads:
            batch.put_file(
                BytesIO(upload.content),
                f"/{upload.relative_path.as_posix()}",
            )
    with output_volume.batch_upload(force=True) as batch:
        batch.put_file(
            BytesIO(prepared.staged_input.content),
            f"/{marker_path}",
        )


def publish_invocation_receipt(
    output_volume: modal.Volume,
    receipt: VolumeUpload,
) -> None:
    """Publish one immutable receipt after its request manifest exists."""
    path = receipt.relative_path.as_posix()
    state = _volume_file_state(output_volume, path, receipt.content)
    if state == "conflict":
        raise RuntimeError(f"Existing invocation receipt conflicts: {path}")
    if state == "match":
        return
    with output_volume.batch_upload(force=True) as batch:
        batch.put_file(BytesIO(receipt.content), f"/{path}")


def _worker_failure(
    batch: tuple[ClaimedSeed, ...],
    error_type: str,
    message: str,
) -> dict[str, object]:
    return {
        "seeds": [item.seed for item in batch],
        "error_type": error_type,
        "message": message,
    }


def _worker_seeds(
    value: object,
    allowed: set[int],
) -> set[int] | None:
    if not isinstance(value, list):
        return None
    seeds: set[int] = set()
    for seed in value:
        if isinstance(seed, bool) or not isinstance(seed, int):
            return None
        seeds.add(seed)
    return seeds if seeds.issubset(allowed) else None


@dataclass(frozen=True, slots=True)
class InProcessInferenceExecutor(InferenceExecutor):
    """Run the established inference functions inside one tracked container."""

    claim_function: Callable[[str, list[int], int], dict[str, object]]
    inspect_function: Callable[[str, list[int], int], list[dict[str, object]]]
    worker_function: Callable[
        [str, str, dict[str, object], list[dict[str, object]]],
        dict[str, object],
    ]
    summary_function: Callable[
        [str, str, dict[str, object]],
        dict[str, object],
    ]
    request_function: Callable[
        [str, str, list[int], list[int], int, str],
        dict[str, object],
    ]

    def claim_seeds(
        self,
        run_id: str,
        seeds: tuple[int, ...],
        *,
        sample_count: int,
    ) -> SeedClaimPlan:
        return seed_claim_plan_from_dict(
            self.claim_function(run_id, list(seeds), sample_count)
        )

    def inspect_seeds(
        self,
        run_id: str,
        seeds: tuple[int, ...],
        *,
        sample_count: int,
    ) -> tuple[dict[str, object], ...]:
        return tuple(self.inspect_function(run_id, list(seeds), sample_count))

    def run_claimed(
        self,
        prepared: PreparedInferenceRun,
        claimed_seeds: tuple[ClaimedSeed, ...],
        *,
        max_workers: int,
        poll_timeout_seconds: int,
    ) -> InferenceBatchOutcome:
        del poll_timeout_seconds
        published: set[int] = set()
        reused: set[int] = set()
        failures: list[dict[str, object]] = []
        for batch in partition_claimed_seeds(claimed_seeds, max_workers):
            try:
                result = self.worker_function(
                    prepared.run_id,
                    prepared.request_id,
                    prepared.staged_input.to_record(),
                    [item.to_dict() for item in batch],
                )
            except Exception as exc:
                failures.append(_worker_failure(batch, type(exc).__name__, str(exc)))
                continue
            allowed = {item.seed for item in batch}
            batch_published = _worker_seeds(result.get("published_seeds"), allowed)
            batch_reused = _worker_seeds(result.get("reused_seeds"), allowed)
            if result.get("run_id") != prepared.run_id or (
                batch_published is None or batch_reused is None
            ):
                failures.append(
                    _worker_failure(batch, "InvalidWorkerResult", repr(result))
                )
                continue
            published.update(batch_published)
            reused.update(batch_reused)
        return InferenceBatchOutcome(
            published_seeds=frozenset(published),
            reused_seeds=frozenset(reused),
            failures=tuple(failures),
        )

    def finalize_summary(
        self,
        prepared: PreparedInferenceRun,
    ) -> dict[str, object]:
        return self.summary_function(
            prepared.run_id,
            prepared.request_id,
            prepared.staged_input.to_record(),
        )

    def finalize_request(
        self,
        prepared: PreparedInferenceRun,
    ) -> dict[str, object]:
        return self.request_function(
            prepared.run_id,
            prepared.request_id,
            list(prepared.submitted_seeds),
            list(prepared.normalized_seeds),
            prepared.sample_count,
            prepared.display_name,
        )
