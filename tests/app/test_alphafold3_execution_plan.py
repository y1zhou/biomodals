"""Pure execution-plan tests for the AlphaFold3 kernel adapter."""

from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold.alphafold3.execution_plan import (
    ALPHAFOLD3_EXECUTION_NODE_KEYS,
    build_alphafold3_execution_plan,
)
from biomodals.app.fold.alphafold3.invocation_cache import prepare_invocation


def _invocation(*, sequence: str = "ACDE", seeds: list[int] | None = None):
    config = AF3Config(
        name="example",
        modelSeeds=seeds or [1, 2],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="A",
                    sequence=sequence,
                )
            )
        ],
    )
    return prepare_invocation(
        config,
        search_msa=True,
        search_protein_templates=True,
        recycle=10,
        sample=5,
    )


def test_alphafold3_execution_plan_preserves_the_fixed_semantic_dag() -> None:
    """The app adapter must model the accepted fixed stage topology."""
    invocation = _invocation()

    plan = build_alphafold3_execution_plan(invocation)

    assert plan.workload_name == "alphafold3"
    assert plan.workload_run_key == invocation.invocation_id
    assert plan.node_keys == ALPHAFOLD3_EXECUTION_NODE_KEYS
    assert plan.terminal_node_keys == ("request-publication",)
    assert {
        node.node_key: tuple(edge.node_key for edge in node.dependencies)
        for node in plan.nodes
    } == {
        "stage-request-input": (),
        "raw-database-searches": ("stage-request-input",),
        "combined-msa-publications": ("raw-database-searches",),
        "protein-template-searches": ("combined-msa-publications",),
        "stage-inference-input": ("protein-template-searches",),
        "seed-predictions": ("stage-inference-input",),
        "inference-summary": ("seed-predictions",),
        "request-publication": ("inference-summary",),
    }
    assert {node.node_key for node in plan.nodes if node.allow_empty_result} == {
        "raw-database-searches",
        "combined-msa-publications",
        "protein-template-searches",
    }


def test_alphafold3_plan_fingerprint_tracks_science_not_worker_limits() -> None:
    """Operational worker limits must not alter scientific compatibility."""
    invocation = _invocation()

    first = build_alphafold3_execution_plan(invocation)
    second = build_alphafold3_execution_plan(invocation)
    changed_sequence = build_alphafold3_execution_plan(_invocation(sequence="FGHI"))
    changed_seeds = build_alphafold3_execution_plan(_invocation(seeds=[3]))

    assert first.workload_plan_fingerprint == second.workload_plan_fingerprint
    assert first.workload_plan_fingerprint != changed_sequence.workload_plan_fingerprint
    assert first.workload_plan_fingerprint != changed_seeds.workload_plan_fingerprint
    assert "max_parallel_search_workers" not in repr(first.scientific_payload)
    assert "max_num_gpus" not in repr(first.scientific_payload)
