"""PPIFlow integration coverage for runtime-discovered kernel Tasks."""

# ruff: noqa: D101, D102, D103, D107

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

from biomodals.execution import (
    DeploymentIdentity,
    NodeAggregationPolicy,
    NodeStatus,
    TaskStatus,
)
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.helper.app_run import volume_app_output
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.workflow.core import NodeRunContext, Workflow, WorkflowNativeNode
from biomodals.workflow.core.runtime import WorkflowRuntime
from biomodals.workflow.ppiflow import manifests
from biomodals.workflow.ppiflow_workflow import (
    LigandMPNNNode,
    PPIFlowPartialNode,
    ReFoldNode,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "PPIFlowWorkflow", 7)


@dataclass
class CandidateSourceNode(WorkflowNativeNode):
    candidate_ids: tuple[str, ...]

    def run(self, context: NodeRunContext) -> AppRunResult:
        if context.volume_root is None or context.workflow_volume_name is None:
            raise RuntimeError("Workflow Volume context is unavailable")
        structures_dir = context.work_dir / "structures"
        structures_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for candidate_id in self.candidate_ids:
            structure = structures_dir / f"{candidate_id}.pdb"
            structure_bytes = f"MODEL {candidate_id}\n".encode()
            structure.write_bytes(structure_bytes)
            rows.append(
                manifests.candidate_manifest_row(
                    candidate_id=candidate_id,
                    stage_name="source",
                    stage_role="filter",
                    operation_mode="retained",
                    candidate_status=AppRunStatus.SUCCEEDED.value,
                    files=[
                        manifests.candidate_file_record(
                            role="structure",
                            workflow_path=structure.relative_to(
                                context.volume_root
                            ).as_posix(),
                            content_sha256=hashlib.sha256(structure_bytes).hexdigest(),
                        )
                    ],
                )
            )
        manifest_path = context.work_dir / manifests.MANIFEST_FILENAME
        manifests.write_manifest(rows, manifest_path)
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                volume_app_output(
                    name="structures",
                    kind=ArtifactKind.STRUCTURES,
                    remote_path=str(structures_dir),
                    mount_root=str(context.volume_root),
                    volume_name=context.workflow_volume_name,
                ),
                manifests.manifest_artifact_output(
                    manifest_path=manifest_path,
                    mount_root=str(context.volume_root),
                    volume_name=context.workflow_volume_name,
                    stage_name="source",
                    row_count=len(rows),
                ),
            ],
        )


class FakeModalDriver:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.results: dict[str, AppRunResult] = {}

    def resolve(self, binding):
        self.events.append(f"resolve:{binding.function_name}")
        return binding.function_name

    def spawn(self, function, *, args, kwargs):
        del function, args
        candidate_id = str(kwargs["candidate_id"])
        self.events.append(f"spawn:{candidate_id}")
        call_id = f"fc-{candidate_id}"
        self.results[call_id] = AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name=f"refolded_{candidate_id}",
                    kind=ArtifactKind.STRUCTURES,
                    storage=InlineBytes(
                        data=f"MODEL {candidate_id}\n".encode(),
                        filename=f"{candidate_id}.pdb",
                        media_type="chemical/x-pdb",
                    ),
                )
            ],
        )
        return call_id

    def observe(self, provider_call_handle_id):
        candidate_id = provider_call_handle_id.removeprefix("fc-")
        self.events.append(f"observe:{candidate_id}")
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=self.results[provider_call_handle_id],
        )

    def cancel(self, provider_call_handle_id):
        del provider_call_handle_id


@pytest.mark.parametrize(
    ("node", "expected_function"),
    [
        (
            ReFoldNode("ReFoldStep", {"run_name": "refold"}),
            "run_ppiflow_refold_candidate",
        ),
        (
            PPIFlowPartialNode("PartialStep", {"run_name": "partial"}),
            "run_ppiflow_partial_candidate",
        ),
        (
            LigandMPNNNode("MPNNStep_stage1", {"run_name": "mpnn"}),
            "run_ppiflow_ligandmpnn_candidate",
        ),
    ],
)
def test_ppiflow_candidates_are_independent_kernel_tasks(
    tmp_path: Path,
    node,
    expected_function: str,
) -> None:
    workflow = Workflow("ppiflow-fanout")
    source = workflow.add_node(
        CandidateSourceNode(("candidate-b", "candidate-a")),
        id="source",
    )
    workflow.add_node(
        node,
        id="candidate-stage",
        inputs={
            "structures": source.outputs(kind=ArtifactKind.STRUCTURES),
            "candidate_manifest": source.outputs(
                kind=ArtifactKind.TABLE,
                role=manifests.MANIFEST_FILE_ROLE,
            ),
        },
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    driver = FakeModalDriver()
    runtime = WorkflowRuntime(
        workflow=workflow,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        volume_root=tmp_path,
        workflow_volume_name="Workflow-outputs",
        modal_driver=driver,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=2,
        now=iter(range(100, 1000)).__next__,
        poll_interval_seconds=0,
    )

    result = runtime.run(workload_run_key="candidate-stage")

    assert result.status == AppRunStatus.SUCCEEDED
    assert driver.events[:4] == [
        f"resolve:{expected_function}",
        "spawn:candidate-b",
        f"resolve:{expected_function}",
        "spawn:candidate-a",
    ]
    tasks = runtime.store.execution.list_tasks(RUN_ID, "candidate-stage")
    assert [task.task_key for task in tasks] == ["candidate-b", "candidate-a"]
    assert tasks[0].scientific_payload["files"][0]["content_sha256"] == (
        hashlib.sha256(b"MODEL candidate-b\n").hexdigest()
    )
    assert [task.status for task in tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
    ]
    assert runtime.store.execution.get_node(RUN_ID, "candidate-stage").status == (
        NodeStatus.SUCCEEDED
    )
    [manifest_artifact] = [
        artifact
        for artifact in runtime.store.artifacts.load_node_output_artifacts(
            "candidate-stage"
        )
        if any(file.role == manifests.MANIFEST_FILE_ROLE for file in artifact.files)
    ]
    frame = manifests.read_manifest(tmp_path / manifest_artifact.storage.path)
    assert frame.get_column("candidate_id").to_list() == [
        "candidate-a",
        "candidate-b",
    ]
