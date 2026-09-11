"""Lean publication accounting and the maximum website result envelope."""

from io import BytesIO
from uuid import UUID

import orjson
import polars as pl

from biomodals.execution.artifacts import materialize_app_run_result
from biomodals.execution.nodes import NodeRunContext
from biomodals.schema import AppRunResult, AppRunStatus
from biomodals.service.humanization.modal import MAX_MANIFEST_BYTES
from biomodals.service.humanization.results import (
    build_humanization_archive,
    query_selection,
)
from biomodals.service.humanization.router import MAX_SELECTION_BYTES
from biomodals.workflow.humanization.artifacts import generated_pairs, json_output
from biomodals.workflow.humanization.contracts import AntibodyPair, CandidateOrigin
from biomodals.workflow.humanization.export import (
    IMGT_MUTATION_SCHEMA,
    export_results,
    generation_table,
)
from biomodals.workflow.humanization.settings import HumanizationSettings
from biomodals.workflow.humanization.tables import candidate_union, selection_table
from biomodals.workflow.humanization.workflow import (
    SCIENTIFIC_VERSIONS,
    HumanizationGenerateNode,
)


def export_context(tmp_path, outcomes):
    """Exercise normal materialized artifact paths, without provider calls."""
    artifacts = materialize_app_run_result(
        result=AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                json_output("generation_outcomes", outcomes),
                json_output("generation_errors", {}),
            ],
        ),
        artifact_volume_name="test",
        result_dir=tmp_path / "generation",
        artifact_dir=tmp_path / "artifacts",
        producing_node_id="generate",
        volume_root=tmp_path,
    ).artifacts
    return NodeRunContext(
        UUID(int=1),
        "test",
        "evaluate",
        "",
        tmp_path / "evaluate",
        tmp_path / "cache",
        {
            "generation_native_test": artifacts,
            "generation_errors": [
                a for a in artifacts if a.source_app_output_name == "generation_errors"
            ],
        },
        tmp_path,
        "test",
    )


def test_hudiff_attempts_retain_rejections_and_duplicate_candidate_links(tmp_path):
    """The ledger preserves native attempts without retaining opaque payloads."""
    parent = AntibodyPair(id="a", vh="ACD", vl="EFG")
    settings = HumanizationSettings(hudiff_ab_candidate_count=3)
    native = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            json_output(
                "native",
                {
                    "schema_version": 1,
                    "pair_result": {
                        "pair_seed": 31337,
                        "candidates": [
                            {
                                "id": "a",
                                "candidate_id": "a__candidate_1",
                                "attempt_index": 1,
                                "vh": "ACH",
                                "vl": "EFG",
                            }
                        ],
                        "attempts": [
                            {"status": "valid"},
                            {
                                "status": "duplicate",
                                "attempt_index": 2,
                                "duplicate_of": "a__candidate_1",
                                "rejection_reason": None,
                                "vh": "ACH",
                                "vl": "EFG",
                            },
                            {
                                "status": "invalid",
                                "attempt_index": 3,
                                "duplicate_of": None,
                                "rejection_reason": "CDR changed",
                            },
                        ],
                    },
                },
            )
        ],
    )
    generated = generated_pairs("hudiff_ab", parent, native)
    result = HumanizationGenerateNode(
        (parent,), settings, "hudiff_ab"
    ).process_remote_task_result(
        "hudiff_ab-0000",
        native,
        {
            "method": "hudiff_ab",
            "parent": parent.model_dump(),
            "parameters": settings.method_arguments("hudiff_ab"),
        },
    )
    outcomes = orjson.loads(
        next(o.storage.data for o in result.outputs if o.name == "generation_outcomes")
    )
    candidates = candidate_union((parent,), generated)
    table = generation_table(
        export_context(tmp_path, outcomes),
        selection_table(candidates, [], []),
        (parent,),
        settings,
        {},
    )
    candidate_id = next(c.candidate_id for c in candidates if not c.is_parent)
    assert table["outcome"].to_list() == ["generated", "duplicate", "rejected"]
    assert table["attempt_index"].to_list() == [1, 2, 3]
    assert table["root_seed"].to_list() == [42, 42, 42]
    assert table["seed"].to_list() == [31337, 31337, 31337]
    assert table["candidate_id"].to_list() == [candidate_id, candidate_id, None]
    assert table["reason"].to_list() == [None, None, "CDR changed"]


def test_maximum_candidate_envelope_has_bounded_verified_manifest(tmp_path):
    """200 parents at maximum yield, ID and chain lengths fit the result reader."""
    settings = HumanizationSettings(
        sapiens_iterations=5, pabnativ2_num_seeds=25, hudiff_ab_candidate_count=25
    )
    parents = tuple(
        AntibodyPair(id="🧬" * 197 + f"{i:03}", vh="A" * 142, vl="C" * 126)
        for i in range(200)
    )
    generated, outcomes = [], []
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for parent in parents:
        index = 0
        for method, count in (
            ("sapiens", 5),
            ("humatch", 1),
            ("pabnativ2", 25),
            ("hudiff_ab", 25),
        ):
            for attempt in range(count):
                vh = "D" * 140 + alphabet[index // 20] + alphabet[index % 20]
                index += 1
                seed = (
                    settings.pabnativ2_seeds[attempt] if method == "pabnativ2" else None
                )
                origin = CandidateOrigin(
                    method=method,
                    source_id=f"{parent.id}_{method}_{attempt}",
                    seed=seed,
                )
                generated.append((parent.id, vh, parent.vl, origin))
                outcomes.append({
                    "parent_id": parent.id,
                    **origin.model_dump(),
                    "iteration": attempt + 1 if method == "sapiens" else None,
                    "outcome": "generated",
                    "vh": vh,
                    "vl": parent.vl,
                })
    candidates = candidate_union(parents, generated)
    assert len(candidates) == 11400
    selection = selection_table(candidates, [], []).with_columns(
        pl.col(pl.Float64).fill_null(1.2345678901234567e-100),
        pl.lit(None, dtype=pl.Int64).alias("quality_tier"),
        pl.lit(None, dtype=pl.Int64).alias("panel_order"),
    )
    context = export_context(tmp_path, outcomes)
    output = export_results(
        context,
        candidates,
        selection,
        pl.DataFrame(schema=IMGT_MUTATION_SCHEMA),
        {},
        {"evaluation": "unavailable"},
        settings,
        SCIENTIFIC_VERSIONS,
        parents,
    )
    root = tmp_path / output.storage.path
    content = (root / "manifest.json").read_bytes()
    assert len(content) < 64 * 1024 < MAX_MANIFEST_BYTES
    manifest = orjson.loads(content)
    assert manifest["candidate_count"] == 11400
    assert (root / "selection.csv").stat().st_size < MAX_SELECTION_BYTES
    page = query_selection(root / "selection.csv", offset=11350)
    assert page.total_rows == 11400
    assert len(page.rows) == 50
    assert {f["path"] for f in manifest["files"]} == {
        "selection.csv",
        "generation.parquet",
        "imgt_mutations.parquet",
    }
    ledger = pl.read_parquet(root / "generation.parquet")
    assert ledger.height == 11200
    assert ledger["candidate_id"].n_unique() == 11200
    assert ledger["candidate_id"].null_count() == 0
    built = build_humanization_archive(root, BytesIO())
    assert built.size_bytes > len(content)
