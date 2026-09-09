"""Parallel paired-antibody humanization and cross-evaluation.

Generates with Sapiens, Humatch, p-AbNatiV2 and HuDiff-Ab, then evaluates
the sequence-distinct union. Research-use candidates require experimental testing.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import modal
import orjson
import polars as pl

from biomodals.app.design.hudiff_ab import app as hudiff_app
from biomodals.app.design.humatch import app as humatch_app
from biomodals.app.design.pabnativ2 import app as pabnativ2_app
from biomodals.app.design.sapiens import app as sapiens_app
from biomodals.execution import (
    COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    CoordinatorNode,
    DeploymentIdentity,
    ExecutionGraph,
    ExecutionOverview,
    NodeRunContext,
    ProviderCallDiagnostic,
    ProviderCallPage,
)
from biomodals.execution.modal import (
    ModalCallDriver,
    development_modal_call_driver,
    execution_coordinator_adapter,
    execution_coordinator_handle,
    execution_coordinator_identity,
    initialize_execution_coordinator_host,
    orchestrator,
    resolve_provider_call_limits,
    stage_execution_launch,
)
from biomodals.execution.model import NodeAggregationPolicy
from biomodals.execution.nodes import ProviderCallSpec, TaskDefinition, TaskProviderNode
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppConfig,
    AppRunResult,
    AppRunStatus,
    ArtifactSelector,
    InlineBytes,
    VolumePath,
)
from biomodals.workflow.display import print_workflow_dag
from biomodals.workflow.humanization.annotation import (
    annotate_humanization_candidate as _annotate_candidate,
)
from biomodals.workflow.humanization.artifacts import (
    archive_members,
    generated_pairs,
    json_output,
)
from biomodals.workflow.humanization.contracts import (
    AntibodyPair,
    CandidateAnnotation,
    CandidateEvaluation,
    CandidateOrigin,
    HumanizationCandidate,
)
from biomodals.workflow.humanization.execution import (
    HumanizationExecutionCoordinator,
    HumanizationExecutionRequest,
    load_execution_request,
    stage_execution_request,
)
from biomodals.workflow.humanization.export import IMGT_MUTATION_SCHEMA, export_results
from biomodals.workflow.humanization.ranking import RANKING_VERSION, rank_panel
from biomodals.workflow.humanization.settings import HumanizationSettings
from biomodals.workflow.humanization.tables import (
    FAMILY_COLUMNS,
    SCORE_COLUMNS,
    candidate_union,
    parse_parents,
    selection_table,
)

METHODS = ("sapiens", "humatch", "pabnativ2", "hudiff_ab")
SCIENTIFIC_VERSIONS = {
    "result_schema": "2",
    "panel_ranking": RANKING_VERSION,
    "biomodals.workflow.humanization": "2",
    "sapiens": sapiens_app.RUNTIME_IDENTITY,
    "sapiens.source": sapiens_app.CONF.repo_commit_hash or "",
    "sapiens.vh": sapiens_app.IDENTITY.vh_revision,
    "sapiens.vl": sapiens_app.IDENTITY.vl_revision,
    "sapiens.tokenizer": sapiens_app.IDENTITY.tokenizer_revision,
    "humatch": humatch_app.RUNTIME_IDENTITY,
    "humatch.assets": "|".join(
        f"{asset.filename}:{asset.md5_hex}" for asset in humatch_app.ASSETS
    ),
    "pabnativ2": pabnativ2_app.SCIENTIFIC_RUNTIME_IDENTITY,
    "pabnativ2.paired_model": pabnativ2_app.PAIRED_MODEL.md5_hex,
    "pabnativ2.structure_model": pabnativ2_app.STRUCTURE_MODEL_ARCHIVE.md5_hex,
    "hudiff_ab": hudiff_app.RUNTIME_IDENTITY,
    "hudiff_ab.model": hudiff_app.ANTIBODY_CHECKPOINT_SHA256,
    "hudiff_ab.patch": hudiff_app.patch_identity(),
    "annotation": "anarci=2020.04.23|hmmer=3.3.2|imgt-boundaries-v1",
}
CONF = AppConfig(
    name="HumanizationWorkflow",
    package_name="biomodals-humanization-workflow",
    version="0.1.0",
    python_version="3.13",
    depends_on_apps=METHODS,
    tags={"depends_on": "-".join(METHODS), "biomodals_tool": "humanization"},
)
OUT_VOLUME = orchestrator.OUT_VOLUME
OUT_VOLUME_NAME = orchestrator.OUT_VOLUME_NAME
OUT_VOLUME_MOUNTPOINT = orchestrator.CONF.output_volume_mountpoint
app = modal.App(CONF.name, image=orchestrator.runtime_image, tags=CONF.tags)
app = include_dependency_apps(app, CONF.depends_on_apps)
annotation_image = hudiff_app.coordinator_image.add_local_python_source(
    "biomodals.workflow.humanization"
)
annotate_humanization_candidate = app.function(
    image=annotation_image, cpu=1, memory=2048, timeout=600
)(_annotate_candidate)


@dataclass
class HumanizationGenerateNode(TaskProviderNode):
    """One method's generation stage with independently owned pair Tasks."""

    parents: tuple[AntibodyPair, ...]
    settings: HumanizationSettings
    method: str

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        """Include a local baseline publication even when every generator fails."""
        tasks = [
            TaskDefinition("parents", [parent.model_dump() for parent in self.parents])
        ]
        method = self.method
        parameters = self.settings.method_arguments(method)
        variants = (
            [dict(parameters, seed=seed) for seed in self.settings.pabnativ2_seeds]
            if method == "pabnativ2"
            else [parameters]
        )
        for index, parent in enumerate(self.parents):
            for replicate, arguments in enumerate(variants):
                suffix = f"-seed-{replicate:02d}" if len(variants) > 1 else ""
                tasks.append(
                    TaskDefinition(
                        f"{method}-{index:04d}{suffix}",
                        {
                            "parent": parent.model_dump(),
                            "method": method,
                            "parameters": arguments,
                        },
                    )
                )
        return tuple(tasks)

    def recover_remote_task_result(
        self, context: NodeRunContext, task: TaskDefinition, expected_fingerprint: str
    ) -> AppRunResult | None:
        """Materialize supplied baselines without spending a provider-call slot."""
        if task.task_key == "parents":
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[json_output("parents", task.scientific_payload)],
            )
        return None

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        """Name existing app operations; never call a child coordinator."""
        payload = task.scientific_payload
        method, parent = payload["method"], payload["parent"]
        kwargs = dict(payload["parameters"])
        if method in {"sapiens", "humatch"}:
            kwargs.update(
                run_name=task.task_key,
                csv_bytes=pl
                .DataFrame([parent])
                .select("id", "vh", "vl")
                .write_csv()
                .encode(),
            )
            operation = f"{method}_humanize"
        else:
            kwargs["pair"] = parent
            operation = f"{method}_humanize_pair"
        return ProviderCallSpec(
            function_name=operation,
            uses_gpu=method in {"pabnativ2", "hudiff_ab"},
            kwargs=kwargs,
            metadata={
                "method": method,
                "parent": parent,
                "parameters": payload["parameters"],
            },
        )

    def process_remote_task_result(
        self, task_key: str, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        """Retain native evidence and add a small normalized candidate publication."""
        result = AppRunResult.model_validate(result)
        if result.status != AppRunStatus.SUCCEEDED:
            return result
        rows = generated_pairs(
            metadata["method"],
            AntibodyPair.model_validate(metadata["parent"]),
            result,
            sapiens_iterations=metadata["parameters"].get("iterations", 1),
        )
        normalized = [
            {"parent_id": parent_id, "vh": vh, "vl": vl, "origin": origin.model_dump()}
            for parent_id, vh, vl, origin in rows
        ]
        return result.model_copy(
            update={"outputs": [*result.outputs, json_output("generated", normalized)]}
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish errors alongside successful siblings, preserving partial status."""
        return AppRunResult(
            status=AppRunStatus.PARTIAL if errors else AppRunStatus.SUCCEEDED,
            outputs=[json_output("generation_errors", dict(errors))],
        )


@dataclass
class HumanizationUnionNode(CoordinatorNode):
    """Deduplicate exact pairs and carry every generating method's provenance."""

    parents: tuple[AntibodyPair, ...]

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Publish the complete union, including unchanged parental baselines."""
        generated = []
        for artifact in (
            artifact
            for name, artifacts in context.inputs.items()
            if name.startswith("generated_")
            for artifact in artifacts
        ):
            rows = orjson.loads(context.resolve_artifact(artifact).read_bytes())
            for row in rows:
                generated.append((
                    row["parent_id"],
                    row["vh"],
                    row["vl"],
                    CandidateOrigin.model_validate(row["origin"]),
                ))
        candidates = candidate_union(self.parents, generated)
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                json_output(
                    "candidate_union",
                    [candidate.model_dump() for candidate in candidates],
                ),
                json_output(
                    "generation_errors",
                    {
                        key: value
                        for name in context.inputs
                        if name.startswith("generation_errors_")
                        for key, value in orjson.loads(
                            context.read_input_bytes(name)
                        ).items()
                    },
                ),
            ],
        )


@dataclass
class HumanizationEvaluateNode(TaskProviderNode):
    """Cross-evaluate every candidate and publish the terminal selection artifacts."""

    parents: tuple[AntibodyPair, ...]
    settings: HumanizationSettings

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        """Use candidate IDs, not generation order, for independently reusable tasks."""
        candidates = orjson.loads(context.read_input_bytes("union"))
        parents = {parent.id: parent.model_dump() for parent in self.parents}
        errors = orjson.loads(context.read_input_bytes("generation_errors"))
        tasks = [
            TaskDefinition("union", candidates),
            TaskDefinition("generation_complete", errors),
        ]
        for candidate in candidates:
            for method in (*SCORE_COLUMNS, "annotation"):
                tasks.append(
                    TaskDefinition(
                        f"{method}-{candidate['candidate_id']}",
                        {
                            "candidate": candidate,
                            "parent": parents[candidate["parent_id"]],
                            "method": method,
                            "vh_target_family": self.settings.humatch_vh_target_family,
                            "vl_target_family": self.settings.humatch_vl_target_family,
                        },
                    )
                )
        return tuple(tasks)

    def recover_remote_task_result(
        self, context: NodeRunContext, task: TaskDefinition, expected_fingerprint: str
    ) -> AppRunResult | None:
        """Publish deterministic local evidence without adding remote wrapper calls."""
        if task.task_key == "union":
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[json_output("evaluated_union", task.scientific_payload)],
            )
        if task.task_key == "generation_complete" and not task.scientific_payload:
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[json_output("generation_complete", True)],
            )
        return None

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        """Prepare native scorers and the common annotation operation under one budget."""
        if task.task_key == "generation_complete":
            raise ValueError(f"Generation incomplete: {task.scientific_payload}")
        payload = task.scientific_payload
        candidate, parent, method = (
            payload["candidate"],
            payload["parent"],
            payload["method"],
        )
        if method == "annotation":
            return ProviderCallSpec(
                function_name="annotate_humanization_candidate",
                uses_gpu=False,
                kwargs={"parent": parent, "candidate": candidate},
                metadata=payload,
            )
        kwargs = {
            "csv_bytes": pl
            .DataFrame([
                {
                    "id": candidate["candidate_id"],
                    "vh": candidate["vh"],
                    "vl": candidate["vl"],
                }
            ])
            .write_csv()
            .encode()
        }
        if method == "humatch":
            kwargs.update(
                reference_vh=parent["vh"],
                reference_vl=parent["vl"],
                vh_target_family=payload["vh_target_family"],
                vl_target_family=payload["vl_target_family"],
            )
        return ProviderCallSpec(
            function_name=f"{method}_score",
            uses_gpu=method == "pabnativ2",
            kwargs=kwargs,
            metadata=payload,
        )

    def process_remote_task_result(
        self, task_key: str, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        """Normalize scalar summaries while retaining the native detailed publication."""
        result = AppRunResult.model_validate(result)
        if (
            result.status != AppRunStatus.SUCCEEDED
            or metadata["method"] == "annotation"
        ):
            return result
        method, candidate = metadata["method"], metadata["candidate"]
        if len(result.outputs) != 1 or not isinstance(
            result.outputs[0].storage, InlineBytes
        ):
            raise ValueError("Scorer must return one bounded inline publication")
        files = archive_members(result.outputs[0].storage.data)
        frame = pl.read_csv(
            BytesIO(files["summary.csv"]), schema_overrides={"id": pl.String}
        )
        if frame.height != 1 or frame["id"].to_list() != [candidate["candidate_id"]]:
            raise ValueError("Scorer returned wrong or missing candidate identity")
        row = frame.row(0, named=True)
        evaluation = CandidateEvaluation(
            parent_id=candidate["parent_id"],
            candidate_id=candidate["candidate_id"],
            evaluator=method,
            status="succeeded",
            scores={name: row[name] for name in SCORE_COLUMNS[method]},
            labels={name: row[name] for name in FAMILY_COLUMNS}
            if method == "humatch"
            else {},
        )
        if any(value is None for value in evaluation.scores.values()):
            raise ValueError("Scorer returned missing summary metrics")
        return result.model_copy(
            update={
                "outputs": [
                    *result.outputs,
                    json_output("evaluation", evaluation.model_dump()),
                ]
            }
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Join every candidate, keeping failures as null metrics and explicit status."""
        candidates = [
            HumanizationCandidate.model_validate(row)
            for row in orjson.loads(context.read_input_bytes("union"))
        ]
        evaluations, annotations, mutation_frames = [], [], []
        for result in results.values():
            for output in result.outputs:
                if output.name not in {"evaluation", "annotation", "imgt_mutations"}:
                    continue
                if context.volume_root is None:
                    raise RuntimeError(
                        "Evaluation aggregation requires execution storage"
                    )
                if (
                    not isinstance(output.storage, VolumePath)
                    or output.storage.volume_name != context.artifact_volume_name
                ):
                    raise ValueError(
                        "Evaluation must be materialized in the execution volume"
                    )
                path = (context.volume_root / output.storage.path).resolve()
                path.relative_to(context.volume_root.resolve())
                if output.name == "imgt_mutations":
                    mutation_frames.append(
                        pl.read_json(path, schema=IMGT_MUTATION_SCHEMA)
                    )
                    continue
                value = orjson.loads(path.read_bytes())
                if output.name == "evaluation":
                    evaluations.append(CandidateEvaluation.model_validate(value))
                elif output.name == "annotation":
                    annotations.append(CandidateAnnotation.model_validate(value))
        for candidate in candidates:
            for method in SCORE_COLUMNS:
                message = errors.get(f"{method}-{candidate.candidate_id}")
                if message:
                    evaluations.append(
                        CandidateEvaluation(
                            parent_id=candidate.parent_id,
                            candidate_id=candidate.candidate_id,
                            evaluator=method,
                            status="failed",
                            error=message,
                        )
                    )
            message = errors.get(f"annotation-{candidate.candidate_id}")
            if message:
                annotations.append(
                    CandidateAnnotation(
                        parent_id=candidate.parent_id,
                        candidate_id=candidate.candidate_id,
                        cdr_preservation="unknown",
                        error=message,
                    )
                )
        mutations = (
            pl.concat(mutation_frames)
            if mutation_frames
            else pl.DataFrame(schema=IMGT_MUTATION_SCHEMA)
        )
        table = rank_panel(
            selection_table(candidates, evaluations, annotations), mutations
        )
        bundle = export_results(
            context,
            candidates,
            table,
            mutations,
            results,
            errors,
            self.settings,
            SCIENTIFIC_VERSIONS,
        )
        return AppRunResult(
            status=AppRunStatus.PARTIAL if errors else AppRunStatus.SUCCEEDED,
            outputs=[bundle],
            metrics={"candidate_count": len(candidates), "failed_tasks": len(errors)},
        )


def build_humanization_workflow(
    csv_bytes: bytes, settings: HumanizationSettings | None = None
) -> ExecutionGraph:
    """Build a barriered generation, union and terminal cross-evaluation graph."""
    return build_humanization_graph(
        parse_parents(csv_bytes), settings or HumanizationSettings()
    )


def validate_humanization_settings(
    settings: HumanizationSettings, *, pair_count: int
) -> None:
    """Apply each app's authoritative native control validation."""
    sapiens_app.validate_parameters(**settings.method_arguments("sapiens"))
    humatch_app.validate_parameters(**settings.method_arguments("humatch"))
    pabnativ2_app.validate_parameters(**settings.method_arguments("pabnativ2"))
    hudiff_app.validate_controls(
        pair_count=pair_count, **settings.method_arguments("hudiff_ab")
    )


def build_humanization_graph(
    parents: tuple[AntibodyPair, ...], settings: HumanizationSettings
) -> ExecutionGraph:
    """Build the same scientific graph from already validated paired inputs."""
    validate_humanization_settings(settings, pair_count=len(parents))
    graph = ExecutionGraph("humanization", scientific_versions=SCIENTIFIC_VERSIONS)
    generation = {
        method: graph.add_node(
            HumanizationGenerateNode(parents, settings, method),
            id=f"generate_{method}",
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
        )
        for method in METHODS
    }
    union = graph.add_node(
        HumanizationUnionNode(parents),
        id="union",
        inputs={
            f"{output}_{method}": ArtifactSelector(
                producing_node_id=node.node_id, pattern=f"{output}.json"
            )
            for method, node in generation.items()
            for output in ("generated", "generation_errors")
        },
        accept_partial_from=list(generation.values()),
        reuse_predecessor_publication=False,
    )
    graph.add_node(
        HumanizationEvaluateNode(parents, settings),
        id="evaluate",
        inputs={
            "union": ArtifactSelector(
                producing_node_id=union.node_id, pattern="candidate_union.json"
            ),
            "generation_errors": ArtifactSelector(
                producing_node_id=union.node_id, pattern="generation_errors.json"
            ),
            **{
                f"generation_native_{method}": ArtifactSelector(
                    producing_node_id=node.node_id
                )
                for method, node in generation.items()
            },
        },
        accept_partial_from=list(generation.values()),
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    return graph


@app.cls(
    cpu=(0.125, 16.125),
    memory=(256, 65536),
    timeout=MAX_TIMEOUT,
    max_containers=1,
    scaledown_window=COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    volumes={OUT_VOLUME_MOUNTPOINT: OUT_VOLUME},
)
@modal.concurrent(max_inputs=8)
class ExecutionCoordinator:
    """One run-scoped writer for both staged API and CLI submissions."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()
    development: bool = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Reload staged inputs before serving the pinned Run."""
        initialize_execution_coordinator_host(self)
        execution_coordinator_identity(self)
        OUT_VOLUME.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionOverview:
        """Drive a staged root Run and return its authoritative overview."""
        return self._adapter(development=development).run()

    @modal.method()
    def resume(self) -> ExecutionOverview:
        """Resume suspended work without retrying conclusive failures."""
        return self._adapter().resume()

    @modal.method()
    def status(self) -> ExecutionOverview:
        """Read the durable kernel projection."""
        return self._adapter().status()

    @modal.method()
    def result(self) -> AppRunResult:
        """Return final artifact locations without transferring scientific files."""
        return self._adapter().result()

    @modal.method()
    def cancel(self) -> ExecutionOverview:
        """Cancel owned work through the shared kernel."""
        return self._adapter().cancel()

    @modal.method()
    def provider_calls(
        self,
        node_key: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        newest_first: bool = False,
    ) -> ProviderCallPage:
        """Page shared provider diagnostics for service progress and logs."""
        return self._adapter().provider_calls(
            cursor=UUID(cursor) if cursor else None,
            limit=limit,
            node_key=node_key,
            newest_first=newest_first,
        )

    @modal.method()
    def provider_call(self, provider_call_id: str) -> ProviderCallDiagnostic | None:
        """Read one provider call without widening the status payload."""
        return self._adapter().provider_call(UUID(provider_call_id))

    @modal.method()
    def prepare_restart(
        self,
        predecessor_execution_run_id: str,
        predecessor_deployment_environment: str,
        predecessor_deployment_name: str,
        predecessor_deployment_version: int,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> None:
        """Prepare a compatible Successor through the shared lifecycle."""
        self._adapter().prepare_restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=DeploymentIdentity(
                predecessor_deployment_environment,
                predecessor_deployment_name,
                predecessor_deployment_version,
            ),
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
        )

    @modal.method()
    def drive_prepared(self) -> ExecutionOverview:
        """Drive a prepared Successor Run."""
        return self._adapter().drive_prepared()

    @modal.method()
    def restart_from(self, predecessor_execution_run_id: str) -> ExecutionOverview:
        """Validate the CLI's staged scientific request against its predecessor."""
        adapter = self._adapter()
        adapter.prepare_restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            candidate_request=load_execution_request(
                OUT_VOLUME_MOUNTPOINT, UUID(self.execution_run_id)
            ),
        )
        return adapter.drive_prepared()

    @modal.exit()
    def exit(self) -> None:
        """Checkpoint the single writer without cancelling provider work."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()

    def _adapter(
        self, *, development: bool | None = None
    ) -> HumanizationExecutionCoordinator:
        execution_run_id, deployment = execution_coordinator_identity(self)
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: HumanizationExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=OUT_VOLUME_MOUNTPOINT,
                output_volume=OUT_VOLUME,
                output_volume_name=OUT_VOLUME_NAME,
                provider_driver=(
                    development_modal_call_driver(
                        {
                            "sapiens_humanize": sapiens_app.sapiens_humanize,
                            "sapiens_score": sapiens_app.sapiens_score,
                            "humatch_humanize": humatch_app.humatch_humanize,
                            "humatch_score": humatch_app.humatch_score,
                            "pabnativ2_humanize_pair": pabnativ2_app.pabnativ2_humanize_pair,
                            "pabnativ2_score": pabnativ2_app.pabnativ2_score,
                            "hudiff_ab_humanize_pair": hudiff_app.hudiff_ab_humanize_pair,
                            "annotate_humanization_candidate": annotate_humanization_candidate,
                        },
                        workload_name=CONF.name,
                    )
                    if selected_mode
                    else ModalCallDriver()
                ),
            ),
        )


@app.local_entrypoint()
def submit_humanization_workflow(
    input_csv: str,
    run_id: str | None = None,
    sapiens_iterations: int = 1,
    sapiens_numbering_scheme: str = "kabat",
    sapiens_cdr_definition: str = "kabat",
    sapiens_mutate_cdrs: bool = False,
    humatch_vh_target_family: str = "auto",
    humatch_vl_target_family: str = "auto",
    humatch_germline_likeness_target: float = 0.4,
    humatch_vh_classifier_target: float = 0.95,
    humatch_vl_classifier_target: float = 0.95,
    humatch_pair_classifier_target: float = 0.95,
    humatch_max_edits: int = 60,
    humatch_mutate_cdrs: bool = False,
    humatch_fixed_vh_positions: str = "",
    humatch_fixed_vl_positions: str = "",
    pabnativ2_mutate_cdrs: bool = False,
    pabnativ2_fixed_vh_positions: str = "",
    pabnativ2_fixed_vl_positions: str = "",
    pabnativ2_residue_score_threshold: float = 0.98,
    pabnativ2_rasa_threshold: float = 0.15,
    pabnativ2_max_relative_pairing_score_decrease: float = 0.1,
    pabnativ2_forbidden_residues: str = "C,M",
    pabnativ2_seed: int = 0,
    pabnativ2_num_seeds: int = 1,
    hudiff_ab_candidate_count: int = 10,
    hudiff_ab_seed: int = 42,
    hudiff_ab_sampling_order: str = "shuffle",
    hudiff_ab_upstream_inference_dropout: bool = True,
    wait: bool = True,
    max_containers: int | None = None,
    max_gpu_containers: int | None = None,
    dry_run: bool = False,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str | None = None,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Generate, cross-evaluate and export humanization candidates from paired CSV.

    Args:
        input_csv: UTF-8 CSV file with exactly id,vh,vl columns.
        run_id: Logical run label; defaults to the input filename stem.
        sapiens_iterations: Greedy passes, 1–5; retain each paired design before deduplication.
        sapiens_numbering_scheme: Native kabat, chothia, or imgt numbering.
        sapiens_cdr_definition: Native kabat, chothia, imgt, or north CDR boundaries.
        sapiens_mutate_cdrs: Permit Sapiens CDR mutation.
        humatch_vh_target_family: Parental heavy-family reference, or auto.
        humatch_vl_target_family: Parental light-family reference, or auto.
        humatch_germline_likeness_target: Native germline-likeness search threshold.
        humatch_vh_classifier_target: Heavy-family search threshold.
        humatch_vl_classifier_target: Light-family search threshold.
        humatch_pair_classifier_target: Native pairing search threshold.
        humatch_max_edits: Maximum native Humatch edit budget.
        humatch_mutate_cdrs: Permit native IMGT CDR mutation in Humatch.
        humatch_fixed_vh_positions: Additional protected heavy IMGT positions.
        humatch_fixed_vl_positions: Additional protected light IMGT positions.
        pabnativ2_mutate_cdrs: Permit native p-AbNatiV2 CDR mutation.
        pabnativ2_fixed_vh_positions: Additional protected heavy AHo positions.
        pabnativ2_fixed_vl_positions: Additional protected light AHo positions.
        pabnativ2_residue_score_threshold: Native residue search threshold.
        pabnativ2_rasa_threshold: Native solvent-accessibility search threshold.
        pabnativ2_max_relative_pairing_score_decrease: Allowed relative pairing decrease.
        pabnativ2_forbidden_residues: Comma-separated forbidden proposed residues.
        pabnativ2_seed: Root seed for native p-AbNatiV2 generation.
        pabnativ2_num_seeds: Independent p-AbNatiV2 runs per parent, 1–25; not guaranteed yield.
        hudiff_ab_candidate_count: Sampling attempts per parent, 1–25; not guaranteed yield.
        hudiff_ab_seed: Root seed for HuDiff generation.
        hudiff_ab_sampling_order: shuffle or left_to_right.
        hudiff_ab_upstream_inference_dropout: Preserve released inference dropout.
        wait: Wait for completion and report output artifact locations.
        max_containers: Global active provider-call ceiling.
        max_gpu_containers: Global GPU subset of the total provider-call ceiling.
        dry_run: Validate and display the DAG without submitting work.
        use_deployed_coordinator: Use an exact deployed workflow version.
        deployment_environment: Modal environment for the containing workflow.
        deployment_name: Containing deployment name; defaults to HumanizationWorkflow.
        deployment_version: Exact numeric deployment version.
        restart_from: Predecessor run UUID for an explicit successor.
    """
    parameters = locals()
    settings = HumanizationSettings.model_validate({
        name: parameters[name] for name in HumanizationSettings.model_fields
    })
    source = Path(input_csv).expanduser().resolve()
    if not source.is_file() or not 0 < source.stat().st_size <= 3 * 1024 * 1024:
        raise ValueError("Input must be a regular CSV file of at most 3 MiB")
    with source.open("rb") as stream:
        content = stream.read(3 * 1024 * 1024 + 1)
    graph = build_humanization_workflow(content, settings)
    total, gpu = resolve_provider_call_limits(
        max_containers=max_containers,
        max_gpu_containers=max_gpu_containers,
        default_max_containers=4,
        default_max_gpu_containers=2,
    )
    if dry_run:
        print_workflow_dag(graph.validate())
        return
    predecessor = UUID(restart_from) if restart_from else None
    if predecessor is not None and not use_deployed_coordinator:
        raise ValueError("Restart requires an exact deployed workflow version")
    execution_run_id = uuid4()
    stage_execution_request(
        OUT_VOLUME,
        execution_run_id,
        HumanizationExecutionRequest(
            run_name=sanitize_filename(run_id or source.stem),
            pairs=parse_parents(content),
            settings=settings,
            max_active_provider_calls=total,
            max_active_gpu_provider_calls=gpu,
        ),
    )
    stage_execution_launch(OUT_VOLUME, execution_run_id, predecessor)
    deployment = DeploymentIdentity(
        deployment_environment if use_deployed_coordinator else "development",
        deployment_name or CONF.name,
        deployment_version if use_deployed_coordinator else 1,
    )
    coordinator = execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=ExecutionCoordinator,
    )
    print(
        f"Deployment Identity: {deployment.environment}/{deployment.deployment_name}/"
        f"v{deployment.deployment_version}\nExecution Run ID: {execution_run_id}",
        flush=True,
    )
    call = (
        coordinator.run.spawn(development=not use_deployed_coordinator)
        if predecessor is None
        else coordinator.restart_from.spawn(
            predecessor_execution_run_id=str(predecessor)
        )
    )
    print(f"Coordinator FunctionCall ID: {call.object_id}", flush=True)
    if wait:
        overview = call.get()
        print(f"Humanization completed: {overview.run.status.value}", flush=True)
        if overview.run.status.value in {"succeeded", "partial"}:
            result = AppRunResult.model_validate(coordinator.result.remote())
            for output in result.outputs:
                if output.name == "humanization_results" and isinstance(
                    output.storage, VolumePath
                ):
                    print(
                        f"humanization_results: volume={output.storage.volume_name} path={output.storage.path}\n"
                        f"Selection table: {output.storage.path}/selection.csv",
                        flush=True,
                    )
