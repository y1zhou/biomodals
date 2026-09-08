"""Iteration retention and deterministic seed fanout without scientific calls."""

import tarfile
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID

import orjson
import polars as pl
import pytest
import zstandard

from biomodals.execution.nodes import NodeRunContext
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.workflow.humanization.artifacts import generated_pairs
from biomodals.workflow.humanization.contracts import AntibodyPair
from biomodals.workflow.humanization.settings import HumanizationSettings
from biomodals.workflow.humanization.tables import candidate_union
from biomodals.workflow.humanization.workflow import (
    HumanizationEvaluateNode,
    HumanizationGenerateNode,
)


@pytest.mark.parametrize("count", [1, 3, 25])
def test_pabnativ2_roots_and_independent_provider_tasks(tmp_path, count):
    """One root is literal; multiple roots are reproducible independent calls."""
    settings = HumanizationSettings(pabnativ2_seed=42, pabnativ2_num_seeds=count)
    roots = settings.pabnativ2_seeds
    assert len(set(roots)) == count
    assert all(0 <= root < 2**32 for root in roots)
    assert (
        roots
        == HumanizationSettings.model_validate_json(
            settings.model_dump_json()
        ).pabnativ2_seeds
    )
    if count == 1:
        assert roots == (42,)
    elif count == 3:
        assert roots == (2746317213, 1181241943, 958682846)
    parents = (
        AntibodyPair(id="a", vh="ACD", vl="EFG"),
        AntibodyPair(id="b", vh="ACD", vl="EFG"),
    )
    context = NodeRunContext(
        UUID(int=1), "test", "generate_pabnativ2", "", tmp_path, tmp_path, {}
    )
    node = HumanizationGenerateNode(parents, settings, "pabnativ2")
    tasks = node.discover_remote_tasks(context)[1:]
    assert len({task.task_key for task in tasks}) == 2 * count
    for parent in parents:
        calls = [
            node.prepare_remote_task(context, task)
            for task in tasks
            if task.scientific_payload["parent"]["id"] == parent.id
        ]
        assert [call.kwargs["seed"] for call in calls] == list(roots)
        assert all(
            call.function_name == "pabnativ2_humanize_pair" and call.uses_gpu
            for call in calls
        )
        assert all(
            call.kwargs
            == {
                **settings.method_arguments("pabnativ2"),
                "seed": root,
                "pair": parent.model_dump(),
            }
            for call, root in zip(calls, roots, strict=True)
        )


@pytest.mark.parametrize("count", [0, -1, 26, True, 1.5, "2"])
def test_pabnativ2_seed_count_rejects_invalid_submission(count):
    """Bound scientific fanout before any provider invocation."""
    with pytest.raises(ValueError):
        HumanizationSettings(pabnativ2_num_seeds=count)


def test_sapiens_iteration_union_evaluates_every_distinct_pair():
    """Retain all pass origins, including unchanged and converged endpoints."""
    parent = AntibodyPair(id="a", vh="ACD", vl="EFG")
    rows = [
        {"id": "a", "iteration": i, "vh": vh, "vl": "EFG"}
        for i, vh in enumerate(("ACD", "ACH", "ACI", "ACI"), start=1)
    ]
    content = pl.DataFrame(rows).write_csv().encode()
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("iteration_designs.csv")
        member.size = len(content)
        archive.addfile(member, BytesIO(content))
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="native",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    data=zstandard.ZstdCompressor().compress(stream.getvalue()),
                    filename="native.zst",
                ),
            )
        ],
    )
    generated = generated_pairs("sapiens", parent, result, sapiens_iterations=4)
    assert len(generated) == 4
    candidates = candidate_union((parent,), generated)
    assert len(candidates) == 3
    converged = next(candidate for candidate in candidates if candidate.vh == "ACI")
    assert {origin.source_id for origin in converged.origins} == {
        "a__iteration_3",
        "a__iteration_4",
    }
    inputs = {
        "union": orjson.dumps([candidate.model_dump() for candidate in candidates]),
        "generation_errors": b"{}",
    }
    node = HumanizationEvaluateNode(
        (parent,), HumanizationSettings(sapiens_iterations=4)
    )
    tasks = node.discover_remote_tasks(
        SimpleNamespace(read_input_bytes=inputs.__getitem__)
    )
    assert {task.task_key for task in tasks} == {
        "union",
        "generation_complete",
        *[
            f"{method}-{candidate.candidate_id}"
            for candidate in candidates
            for method in ("sapiens", "humatch", "pabnativ2", "annotation")
        ],
    }
    with pytest.raises(ValueError, match="parent"):
        generated_pairs("sapiens", parent, result, sapiens_iterations=3)
