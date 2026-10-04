"""Exercise RNA request/plan transport without importing Modal app entrypoints.

Run with the target image interpreter and inherited helper dependencies. The
report and request bytes are retained for inspection by CI or a local reviewer.
"""

from __future__ import annotations

import argparse
import importlib.abc
import pickle
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import orjson


class BlockEntrypoints(importlib.abc.MetaPathFinder):
    """Make accidental request-decoding imports fail as they would in a lean image."""

    def find_spec(self, fullname, path=None, target=None):
        """Allow runtime dependencies, but no scientific app composition roots."""
        if fullname.endswith(("oligoformer_app", "ensirna_app")):
            raise ImportError(f"Request transport imported entrypoint {fullname}")
        return None


def main() -> None:
    """Round-trip actual requests and worker plans and retain a versioned report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.meta_path.insert(0, BlockEntrypoints())
    from biomodals.app.score.ensirna_contracts import EnsirnaPreparationPlan
    from biomodals.app.score.ensirna_execution import EnsirnaExecutionRequest
    from biomodals.app.score.oligoformer_contracts import (
        DEFAULT_EXECUTION_CONFIG,
        OligoformerRunConfig,
        OligoformerRunPlan,
    )
    from biomodals.app.score.oligoformer_execution import (
        OligoformerExecutionRequest,
        _run_plan_from_value,
    )

    fasta = b">target\nAUGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGC\n"
    request = OligoformerExecutionRequest(
        run_name="runtime-check",
        mrna_fasta_bytes=fasta,
        sirna_fasta_bytes=None,
        utr_bytes=None,
        orf_bytes=None,
        targetscan_ref_shard_size=None,
        force=False,
        force_generation=None,
        app_version="pinned",
        model_version="fixture",
        reference_version=None,
        **asdict(OligoformerRunConfig()),
        **asdict(DEFAULT_EXECUTION_CONFIG),
    )
    ensirna = EnsirnaExecutionRequest(
        run_name="runtime-check",
        fasta_content=fasta,
        prepare_workers=1,
        pdb_cores=1,
        preprocess_shard_size=1,
        force_generation=None,
        app_version="pinned",
    )
    reports = {}
    for name, value in (("oligoformer", request), ("ensirna", ensirna)):
        content = value.to_bytes()
        (args.output / f"{name}-request.json").write_bytes(content)
        restored = type(value).from_bytes(content)
        if restored != value or restored.execution_plan != value.execution_plan:
            raise RuntimeError(f"{name} changed across request transport")
        reports[name] = {
            "request_sha256": sha256(content).hexdigest(),
            "plan_fingerprint": restored.execution_plan.workload_plan_fingerprint,
        }
    plan = OligoformerRunPlan(
        cache_key="cache",
        efficacy_key="efficacy",
        run_root="/output/cache",
        efficacy_dir="/output/efficacy",
        output_dir="/output/final",
        output_stems=("target",),
        config=OligoformerRunConfig(),
        postprocess_key="final",
        efficacy_ready=False,
        evidence_ready=False,
        final_ready=False,
    )
    if _run_plan_from_value(orjson.loads(orjson.dumps(asdict(plan)))) != plan:
        raise RuntimeError("OligoFormer plan changed across JSON transport")
    for value in (
        plan,
        EnsirnaPreparationPlan(
            "cache", "/prepared", "/input.json", "/processed", 1, 0, [], False
        ),
    ):
        # Trusted, locally created fixture; no external pickle is accepted.
        if pickle.loads(pickle.dumps(value)) != value:  # noqa: S301
            raise RuntimeError("Worker plan changed across pickle transport")
    report = {
        "python": sys.version,
        "requests": reports,
        "plan_transport": "passed",
        "entrypoint_imports": "blocked",
    }
    (args.output / "validation.json").write_bytes(
        orjson.dumps(report, option=orjson.OPT_INDENT_2)
    )


if __name__ == "__main__":
    main()
