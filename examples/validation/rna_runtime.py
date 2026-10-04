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
    parser.add_argument("app", choices=("oligoformer", "ensirna"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.meta_path.insert(0, BlockEntrypoints())
    fasta = b">target\nAUGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGC\n"
    if args.app == "oligoformer":
        from biomodals.app.score.oligoformer_contracts import (
            DEFAULT_EXECUTION_CONFIG,
            OligoformerRunConfig,
            OligoformerRunPlan,
        )
        from biomodals.app.score.oligoformer_execution import (
            OligoformerExecutionRequest,
            _run_plan_from_value,
        )

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
    else:
        from biomodals.app.score.ensirna_contracts import EnsirnaPreparationPlan
        from biomodals.app.score.ensirna_execution import EnsirnaExecutionRequest

        request = EnsirnaExecutionRequest(
            run_name="runtime-check",
            fasta_content=fasta,
            prepare_workers=1,
            pdb_cores=1,
            preprocess_shard_size=1,
            force_generation=None,
            app_version="pinned",
        )
        plan = EnsirnaPreparationPlan(
            "cache", "/prepared", "/input.json", "/processed", 1, 0, [], False
        )
    content = request.to_bytes()
    (args.output / f"{args.app}-request.json").write_bytes(content)
    restored = type(request).from_bytes(content)
    if restored != request or restored.execution_plan != request.execution_plan:
        raise RuntimeError(f"{args.app} changed across request transport")
    # Trusted, locally created fixture; no external pickle is accepted.
    if pickle.loads(pickle.dumps(plan)) != plan:  # noqa: S301
        raise RuntimeError("Worker plan changed across pickle transport")
    report = {
        "python": sys.version,
        "app": args.app,
        "request_sha256": sha256(content).hexdigest(),
        "plan_fingerprint": restored.execution_plan.workload_plan_fingerprint,
        "plan_transport": "passed",
        "entrypoint_imports": "blocked",
    }
    (args.output / "validation.json").write_bytes(
        orjson.dumps(report, option=orjson.OPT_INDENT_2)
    )


if __name__ == "__main__":
    main()
