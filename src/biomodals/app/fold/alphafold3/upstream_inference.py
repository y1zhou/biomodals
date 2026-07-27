"""Pinned upstream AlphaFold 3 inference-process integration.

The Modal app supplies mounted paths and publication state. This module owns
the upstream command shape, worker input construction, and summary data JSON
construction so the composition root does not duplicate AlphaFold semantics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.fold.alphafold3.inference_inputs import (
    serialize_af3_input,
    validate_inference_parameters,
    validate_upstream_af3_input,
)
from biomodals.app.fold.alphafold3.input_enrichment import (
    fill_missing_msa_for_inference,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
    InferenceRuntime,
    SeedWorkerTask,
    canonical_output_name,
    claimed_seed_from_dict,
    finalize_run_summary,
    run_seed_prediction_worker,
)
from biomodals.helper.shell import run_command


@dataclass(frozen=True, slots=True)
class UpstreamInferenceRuntime:
    """Mounted paths and durable publication state for one GPU worker."""

    predictions: InferenceRuntime
    source_root: Path
    model_root: Path
    jax_cache_dir: Path


def run_upstream_seed_worker(
    runtime: UpstreamInferenceRuntime,
    json_bytes: bytes,
    run_id: str,
    recycle: int,
    sample_count: int,
    claimed_seed_records: list[dict[str, object]],
) -> dict[str, object]:
    """Run and marker-last publish one disjoint upstream seed group."""
    validate_inference_parameters(recycle, sample_count)
    claimed_seeds = tuple(
        claimed_seed_from_dict(record) for record in claimed_seed_records
    )
    base_config = validate_upstream_af3_input(AF3Config.model_validate_json(json_bytes))

    def execute(
        worker_root: Path,
        canonical_name: str,
        seeds: tuple[int, ...],
    ) -> None:
        config = base_config.model_copy(deep=True)
        config.name = canonical_name
        config.modelSeeds = list(seeds)
        config = fill_missing_msa_for_inference(config)
        input_json_path = worker_root / "input.json"
        input_json_path.write_bytes(serialize_af3_input(config))
        print(f"💊 Running inference for {canonical_name} with seeds {list(seeds)}")
        run_command(
            [
                sys.executable,
                str(runtime.source_root / "run_alphafold.py"),
                "--run_inference=true",
                "--run_data_pipeline=false",
                f"--json_path={input_json_path}",
                f"--output_dir={worker_root}",
                f"--model_dir={runtime.model_root}",
                f"--jax_compilation_cache_dir={runtime.jax_cache_dir}",
                f"--num_recycles={recycle}",
                f"--num_diffusion_samples={sample_count}",
            ],
            output_mode="tee",
            log_file=worker_root / "run.log",
        )

    return run_seed_prediction_worker(
        runtime.predictions,
        SeedWorkerTask(
            run_id=run_id,
            sample_count=sample_count,
            claimed_seeds=claimed_seeds,
        ),
        execute,
    )


def finalize_upstream_run_summary(
    runtime: InferenceRuntime,
    json_bytes: bytes,
    run_id: str,
    sample_count: int,
) -> dict[str, object]:
    """Rebuild the accumulated summary using upstream's data JSON format."""
    from alphafold3.common import (  # type: ignore[ty:unresolved-import]
        folding_input,
    )

    base_config = validate_upstream_af3_input(AF3Config.model_validate_json(json_bytes))

    def build_data_json(seeds: tuple[int, ...]) -> bytes:
        config = base_config.model_copy(deep=True)
        config.name = canonical_output_name(run_id)
        config.modelSeeds = list(seeds)
        fold_input = folding_input.Input.from_json(serialize_af3_input(config).decode())
        return fold_input.to_json().encode()

    return finalize_run_summary(
        runtime,
        run_id,
        sample_count=sample_count,
        build_data_json=build_data_json,
    )
