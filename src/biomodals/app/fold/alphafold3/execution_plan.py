"""AlphaFold3's fixed semantic DAG for the shared execution kernel."""

from biomodals.app.fold.alphafold3.inference_inputs import (
    ALPHAFOLD3_APP_VERSION,
    hash_sequences,
)
from biomodals.app.fold.alphafold3.invocation_cache import (
    INVOCATION_IDENTITY_SCHEMA,
    PreparedInvocation,
)
from biomodals.app.fold.alphafold3.profiles import ALPHAFOLD3_COMMIT
from biomodals.execution import ExecutionPlan, NodeDependency, NodePlan

ALPHAFOLD3_EXECUTION_NODE_KEYS = (
    "stage-request-input",
    "raw-database-searches",
    "combined-msa-publications",
    "protein-template-searches",
    "stage-inference-input",
    "seed-predictions",
    "inference-summary",
    "request-publication",
)


def build_alphafold3_execution_plan(
    invocation: PreparedInvocation,
) -> ExecutionPlan:
    """Bind the existing invocation identity to AlphaFold3's semantic DAG."""
    if invocation.identity.get(
        "schema"
    ) != INVOCATION_IDENTITY_SCHEMA or invocation.invocation_id != hash_sequences(
        invocation.identity
    ):
        raise ValueError("Prepared AlphaFold3 invocation identity is invalid")

    nodes: list[NodePlan] = []
    previous: str | None = None
    empty_result_nodes = {
        "raw-database-searches",
        "combined-msa-publications",
        "protein-template-searches",
    }
    for node_key in ALPHAFOLD3_EXECUTION_NODE_KEYS:
        nodes.append(
            NodePlan(
                node_key=node_key,
                dependencies=(
                    (NodeDependency(previous),) if previous is not None else ()
                ),
                allow_empty_result=node_key in empty_result_nodes,
            )
        )
        previous = node_key

    return ExecutionPlan(
        workload_name="alphafold3",
        workload_run_key=invocation.invocation_id,
        nodes=tuple(nodes),
        scientific_payload=invocation.identity,
        scientific_versions={
            "alphafold3_app": ALPHAFOLD3_APP_VERSION,
            "alphafold3_upstream": ALPHAFOLD3_COMMIT,
            "invocation_identity": INVOCATION_IDENTITY_SCHEMA,
        },
    )
