"""Pure graph-priority Provider Call admission tests."""

# ruff: noqa: D103

from biomodals.execution import (
    ExecutionPlan,
    NodeDependency,
    NodePlan,
    ProviderBinding,
)
from biomodals.execution.scheduler import (
    ProviderCallCandidate,
    required_node_ranks,
    select_admissible_candidates,
)

GPU_X = ProviderBinding("prod", "app", 1, "gpu-x", True, "image-x")
GPU_Y = ProviderBinding("prod", "app", 1, "gpu-y", True, "image-y")
CPU_X = ProviderBinding("prod", "app", 1, "cpu-x", False, "image-x")


def _candidate(
    key: str,
    ordinal: int,
    *,
    binding: ProviderBinding,
    depth: int = 2,
    unblocking_span: int = 1,
) -> ProviderCallCandidate:
    return ProviderCallCandidate(
        candidate_key=key,
        node_key="work",
        node_ordinal=0,
        task_keys=(key,),
        task_ordinal=ordinal,
        binding=binding,
        compatibility_key="same",
        depth=depth,
        unblocking_span=unblocking_span,
    )


def test_required_node_rank_uses_depth_and_unfinished_descendant_span() -> None:
    plan = ExecutionPlan(
        workload_name="branched",
        nodes=(
            NodePlan(node_key="source"),
            NodePlan(
                node_key="left",
                dependencies=(NodeDependency(node_key="source"),),
            ),
            NodePlan(
                node_key="right",
                dependencies=(NodeDependency(node_key="source"),),
            ),
            NodePlan(
                node_key="result",
                dependencies=(
                    NodeDependency(node_key="left"),
                    NodeDependency(node_key="right"),
                ),
            ),
        ),
    )

    ranks = required_node_ranks(
        plan,
        required_node_keys={"source", "left", "right", "result"},
        unfinished_node_keys={"source", "right", "result"},
    )

    assert ranks["source"].depth == 0
    assert ranks["source"].unblocking_span == 2
    assert ranks["left"].depth == 1
    assert ranks["left"].unblocking_span == 1
    assert ranks["result"].depth == 2
    assert ranks["result"].unblocking_span == 0


def test_admission_keeps_graph_rank_primary_then_gpu_and_image_cohorts() -> None:
    candidates = (
        _candidate("cpu-critical", 0, binding=CPU_X, depth=3),
        _candidate("gpu-x-1", 1, binding=GPU_X),
        _candidate("gpu-y-1", 2, binding=GPU_Y),
        _candidate("gpu-x-2", 3, binding=GPU_X),
        _candidate("cpu-x", 4, binding=CPU_X),
    )

    selected = select_admissible_candidates(
        candidates,
        available_total_slots=5,
        available_gpu_slots=3,
    )

    assert [candidate.candidate_key for candidate in selected] == [
        "cpu-critical",
        "gpu-x-1",
        "gpu-x-2",
        "gpu-y-1",
        "cpu-x",
    ]


def test_one_cycle_fills_feasible_slots_and_gpu_saturation_skips_only_gpu() -> None:
    candidates = (
        _candidate("gpu-0", 0, binding=GPU_X),
        _candidate("gpu-1", 1, binding=GPU_X),
        *(_candidate(f"cpu-{index}", index + 2, binding=CPU_X) for index in range(10)),
    )

    selected = select_admissible_candidates(
        candidates,
        available_total_slots=5,
        available_gpu_slots=2,
    )
    selected_without_gpu_capacity = select_admissible_candidates(
        candidates,
        available_total_slots=3,
        available_gpu_slots=0,
    )

    assert [candidate.candidate_key for candidate in selected] == [
        "gpu-0",
        "gpu-1",
        "cpu-0",
        "cpu-1",
        "cpu-2",
    ]
    assert [candidate.candidate_key for candidate in selected_without_gpu_capacity] == [
        "cpu-0",
        "cpu-1",
        "cpu-2",
    ]
