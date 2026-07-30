"""Pure execution-plan tests for the AlphaFold3 kernel adapter."""

from pathlib import PurePosixPath

from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold.alphafold3.execution_plan import (
    ALPHAFOLD3_EXECUTION_NODE_KEYS,
    build_alphafold3_execution_plan,
)
from biomodals.app.fold.alphafold3.execution_tasks import (
    combined_msa_task_plan,
    raw_search_task_plan,
    seed_prediction_task_plan,
    template_search_task_plan,
)
from biomodals.app.fold.alphafold3.inference_inputs import prepare_inference_run
from biomodals.app.fold.alphafold3.invocation_cache import prepare_invocation
from biomodals.app.fold.alphafold3.msa_search import (
    MsaArtifactReference,
    MsaAssemblyTask,
    RawSearchTask,
    sequence_hash,
)
from biomodals.app.fold.alphafold3.template_search import TemplateTask


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


def test_alphafold3_search_tasks_bind_existing_scientific_identities() -> None:
    """Task fingerprints should reuse cache identities rather than paths."""
    raw = RawSearchTask(database_id="uniref90", sequence="ACDE")
    raw_plan = raw_search_task_plan(raw, search_identity="a" * 64)
    assembly = MsaAssemblyTask(
        polymer="protein",
        sequence="ACDE",
        include_unpaired=True,
        include_paired=True,
    )
    assembly_plan = combined_msa_task_plan(
        assembly,
        raw_publication_digests={
            "uniref90": "b" * 64,
            "uniprot": "c" * 64,
        },
    )
    reference = MsaArtifactReference.from_content(
        relative_path=PurePosixPath("Protein/cache/unpaired.a3m"),
        content=b">query\nACDE\n",
    )
    template = TemplateTask(
        sequence="ACDE",
        unpaired_msa=None,
        unpaired_msa_reference=reference,
        publish_canonical=True,
    )
    template_plan = template_search_task_plan(template)

    assert raw_plan.task_key == f"uniref90:{sequence_hash('ACDE')}:{'a' * 64}"
    assert raw_plan.scientific_payload["search_identity"] == "a" * 64
    assert assembly_plan.scientific_payload["raw_publication_digests"] == {
        "uniprot": "c" * 64,
        "uniref90": "b" * 64,
    }
    assert template_plan.scientific_payload["template_identity"] == (
        template.template_identity
    )
    for plan in (raw_plan, assembly_plan, template_plan):
        assert "path" not in repr(plan.scientific_payload).lower()
        assert "max_parallel" not in repr(plan.scientific_payload)


def test_alphafold3_seed_tasks_are_independent_of_gpu_partitioning() -> None:
    """One seed remains one scientific Task regardless of worker grouping."""
    config = AF3Config(
        name="example",
        modelSeeds=[3, 1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="A",
                    sequence="ACDE",
                    unpairedMsa="",
                    pairedMsa="",
                    templates=[],
                )
            )
        ],
    )
    prepared = prepare_inference_run(config, recycle=10, sample=5)

    plans = tuple(
        seed_prediction_task_plan(prepared, seed) for seed in prepared.normalized_seeds
    )

    assert tuple(plan.task_key for plan in plans) == ("seed:1", "seed:3")
    assert {plan.scientific_payload["run_id"] for plan in plans} == {prepared.run_id}
    assert all("max_num_gpus" not in repr(plan.scientific_payload) for plan in plans)
