"""Parallel VHH-oriented generation, exact unions and independent dual scoring."""

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import modal
import orjson
import polars as pl

from biomodals.app.design.abnativ2_vhh.models import (
    MODELS,
    NBFORGE_SHA256,
)
from biomodals.app.design.abnativ2_vhh.models import (
    RUNTIME_IDENTITY as ABNATIV_IDENTITY,
)
from biomodals.app.design.hudiff_nb.patches import patch_identity
from biomodals.app.design.hudiff_nb.worker import CHECKPOINT
from biomodals.app.design.hudiff_nb.worker import RUNTIME_IDENTITY as HUDIFF_IDENTITY
from biomodals.execution import (
    COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    CoordinatorNode,
    DeploymentIdentity,
    ExecutionGraph,
    ExecutionOverview,
    ProviderCallDiagnostic,
    ProviderCallPage,
    republish_execution_artifact,
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
from biomodals.execution.nodes import (
    NodeRunContext,
    ProviderCallSpec,
    TaskDefinition,
    TaskProviderNode,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.antibody import (
    ARPEGGIA_VERSION,
    BIOPYTHON_VERSION,
    GERMLINE_REFERENCE,
)
from biomodals.helper.antibody_tables import GERMLINE_TABLE_SCHEMA
from biomodals.helper.archives import flat_archive_members
from biomodals.helper.artifacts import read_bounded_file_bytes
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppConfig,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    ArtifactSelector,
    InlineBytes,
    VolumePath,
)
from biomodals.workflow.display import print_workflow_dag
from biomodals.workflow.nanobody_humanization.annotation import (
    annotate_nanobody_candidates as _annotate_candidates,
)
from biomodals.workflow.nanobody_humanization.artifacts import (
    generation_frame,
    json_output,
    output_bytes,
    score_tables,
    table_output,
)
from biomodals.workflow.nanobody_humanization.execution import (
    NanobodyExecutionCoordinator,
    NanobodyExecutionRequest,
    load_execution_request,
    stage_execution_request,
)
from biomodals.workflow.nanobody_humanization.export import export_results
from biomodals.workflow.nanobody_humanization.preparation import (
    PREPARATION_VERSION,
    PreparedVH,
    VHInput,
    prepare_batch,
)
from biomodals.workflow.nanobody_humanization.ranking import rank_panel
from biomodals.workflow.nanobody_humanization.settings import NanobodySettings
from biomodals.workflow.nanobody_humanization.tables import (
    GENERATION_SCHEMA,
    KEY,
    add_scores,
    candidate_union,
    mutation_table,
)

METHODS = ("abnativ2_vhh", "hudiff_nb")
SCIENTIFIC_VERSIONS = {
    "result_schema": "1",
    "workflow": "1",
    "preparation": PREPARATION_VERSION,
    "panel_ranking": "1",
    "abnativ2_vhh": ABNATIV_IDENTITY,
    "abnativ2_vhh.models": "|".join(
        f"{model.filename}:{model.md5_hex}" for model in MODELS
    ),
    "nbforge.checkpoint": NBFORGE_SHA256,
    "hudiff_nb": HUDIFF_IDENTITY,
    "hudiff_nb.checkpoint": CHECKPOINT.sha256,
    "hudiff_nb.patch": patch_identity(),
    "sequence_metrics": f"biopython={BIOPYTHON_VERSION}|full-input-pi-v1",
    "germlines": f"arpeggia={ARPEGGIA_VERSION}|{GERMLINE_REFERENCE}|all-species-v1",
}

CONF = AppConfig(
    name="NanobodyHumanizationWorkflow",
    package_name="biomodals-nanobody-humanization-workflow",
    version="1.0.0",
    python_version="3.13",
    depends_on_apps=METHODS,
    tags={"depends_on": "-".join(METHODS), "biomodals_tool": "nanobody_humanization"},
)
OUT_VOLUME = orchestrator.OUT_VOLUME
OUT_VOLUME_NAME = orchestrator.OUT_VOLUME_NAME
OUT_VOLUME_MOUNTPOINT = orchestrator.CONF.output_volume_mountpoint
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .uv_pip_install(f"biopython=={BIOPYTHON_VERSION}")
    .pipe(patch_image_for_helper, include_workflow_modules=True)
)
app = include_dependency_apps(
    modal.App(CONF.name, image=runtime_image, tags=CONF.tags), METHODS
)
annotation_image = (
    modal.Image
    .debian_slim(python_version="3.13")
    .uv_pip_install(f"arpeggia=={ARPEGGIA_VERSION}", f"biopython=={BIOPYTHON_VERSION}")
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.workflow.nanobody_humanization")
)
annotate_nanobody_candidates = app.function(
    image=annotation_image, cpu=2, memory=4096, timeout=3600
)(_annotate_candidates)


@dataclass
class NanobodyGenerateNode(TaskProviderNode):
    """One native generator, independently owned per-parent calls."""

    parents: tuple[PreparedVH, ...]
    settings: NanobodySettings
    method: str

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        """Retain a baseline so one failed arm does not block the other arm's union."""
        return (
            TaskDefinition("parents", [parent.model_dump() for parent in self.parents]),
            *(
                TaskDefinition(f"parent-{index:04d}", parent.model_dump())
                for index, parent in enumerate(self.parents)
            ),
        )

    def recover_remote_task_result(
        self, context: NodeRunContext, task: TaskDefinition, expected_fingerprint: str
    ) -> AppRunResult | None:
        """Publish frozen baselines locally without calling a model."""
        if task.task_key == "parents":
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[json_output("parents", task.scientific_payload)],
            )
        return None

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        """Pass one actual VH and the frozen policy; no dummy light chain."""
        parent = task.scientific_payload
        kwargs = {
            "parent": {
                "sequence": parent["sequence"],
                "protected_indices": parent["protected_indices"],
            }
        }
        if self.method == "hudiff_nb":
            kwargs.update(
                seed=self.settings.root_seed,
                candidate_count=self.settings.hudiff_nb_candidate_count,
            )
        else:
            kwargs["settings"] = {
                key.removeprefix("abnativ2_"): value
                for key, value in self.settings.model_dump().items()
                if key.startswith("abnativ2_")
            }
        return ProviderCallSpec(
            function_name=f"{self.method}_humanize",
            uses_gpu=True,
            kwargs=kwargs,
            metadata={"parent": parent},
        )

    def process_remote_task_result(
        self, task_key: str, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        """Reject mismatched or policy-violating native reports before publication."""
        result = AppRunResult.model_validate(result)
        if result.status == AppRunStatus.SUCCEEDED:
            reports = [
                output for output in result.outputs if output.name == "generation"
            ]
            if len(reports) != 1 or not isinstance(reports[0].storage, InlineBytes):
                raise ValueError("Expected one native generation report")
            generation_frame(
                PreparedVH.model_validate(metadata["parent"]),
                self.method,
                self.settings,
                orjson.loads(reports[0].storage.data),
            )
        return result

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Consolidate attempts and failures without copying native structures."""
        frames = []
        for index, parent in enumerate(self.parents):
            key = f"parent-{index:04d}"
            if key in errors:
                frames.append(
                    pl.DataFrame(
                        [
                            {
                                "parent_id": parent.id,
                                "method": self.method,
                                "error": errors[key],
                                "seed": self.settings.root_seed
                                if self.method == "hudiff_nb"
                                else None,
                            }
                        ],
                        schema=GENERATION_SCHEMA,
                    )
                )
            else:
                output = next(
                    output
                    for output in results[key].outputs
                    if output.name == "generation"
                )
                frames.append(
                    generation_frame(
                        parent,
                        self.method,
                        self.settings,
                        orjson.loads(output_bytes(context, output)),
                    )
                )
        return AppRunResult(
            status=AppRunStatus.PARTIAL if errors else AppRunStatus.SUCCEEDED,
            outputs=[table_output(context, "generation", pl.concat(frames))],
        )


@dataclass
class NanobodyUnionNode(CoordinatorNode):
    """Exact within-parent union; all failed generators stop before evaluation."""

    parents: tuple[PreparedVH, ...]

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Recheck every mutation against the saved prepared-parent policy."""
        generation = pl.concat([
            pl.read_parquet(context.resolve_artifact(context.single_input(method)))
            for method in METHODS
        ])
        if not generation.filter(
            pl.col("error").is_null() & pl.col("vh").is_not_null()
        ).height:
            raise ValueError(
                "Every generator failed; no usable nanobody generation completed"
            )
        candidates, origins = candidate_union(self.parents, generation)
        mutations = mutation_table(candidates, self.parents)
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                table_output(context, "candidates", candidates),
                table_output(context, "generation", origins),
                table_output(context, "mutations", mutations),
            ],
        )


@dataclass
class NanobodyEvaluateNode(TaskProviderNode):
    """Bounded per-parent VH2/VHH2 batches plus unique-sequence CPU annotations."""

    parents: tuple[PreparedVH, ...]
    settings: NanobodySettings

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        """Only tiny provider payloads become Python objects; wide tables stay Polars."""
        candidates = pl.read_parquet(
            context.resolve_artifact(context.single_input("candidates"))
        )
        tasks = [
            TaskDefinition("baseline", candidates["candidate_id"].to_list()),
            TaskDefinition("annotation", candidates.select(*KEY, "vh").write_csv()),
        ]
        for index, parent in enumerate(self.parents):
            sequences = (
                candidates
                .filter(pl.col("parent_id") == parent.id)
                .select(
                    pl.col("candidate_id").alias("id"), pl.col("vh").alias("sequence")
                )
                .to_dicts()
            )
            for model in ("VH2", "VHH2"):
                tasks.append(
                    TaskDefinition(
                        f"{model}-{index:04d}",
                        {"sequences": sequences, "model_type": model},
                    )
                )
        return tuple(tasks)

    def recover_remote_task_result(
        self, context: NodeRunContext, task: TaskDefinition, expected_fingerprint: str
    ) -> AppRunResult | None:
        """Keep the union usable even when every evaluation is unavailable."""
        if task.task_key == "baseline":
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[json_output("baseline", task.scientific_payload)],
            )
        return None

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        """One included model operation per bounded parent/model batch."""
        if task.task_key == "annotation":
            return ProviderCallSpec(
                function_name="annotate_nanobody_candidates",
                uses_gpu=False,
                kwargs={"csv_bytes": task.scientific_payload.encode()},
                metadata={"csv": task.scientific_payload},
            )
        return ProviderCallSpec(
            function_name="abnativ2_vhh_score",
            uses_gpu=True,
            kwargs=task.scientific_payload,
            metadata=task.scientific_payload,
        )

    def process_remote_task_result(
        self, task_key: str, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        """Bind score summaries and detailed rows to their exact submitted sequences."""
        result = AppRunResult.model_validate(result)
        if result.status != AppRunStatus.SUCCEEDED:
            return result
        if len(result.outputs) != 1 or not isinstance(
            result.outputs[0].storage, InlineBytes
        ):
            raise ValueError("Expected one bounded scientific table archive")
        tables = {
            name: pl.read_parquet(BytesIO(data))
            for name, data in flat_archive_members(
                result.outputs[0].storage.data
            ).items()
        }
        if task_key == "annotation":
            expected = pl.read_csv(
                BytesIO(metadata["csv"].encode()), infer_schema=False
            )
            if set(tables) != {"annotations.parquet", "germlines.parquet"}:
                raise ValueError("Unexpected annotation tables")
            for frame in tables.values():
                if (
                    not frame
                    .select(KEY)
                    .sort(KEY)
                    .equals(expected.select(KEY).sort(KEY))
                ):
                    raise ValueError("Annotation changed candidate identity")
        else:
            if (
                not {"summary.parquet"}
                <= tables.keys()
                <= {
                    "summary.parquet",
                    "native_scores.parquet",
                    "residue_scores.parquet",
                }
            ):
                raise ValueError("Unexpected scorer tables")
            expected = pl.DataFrame(metadata["sequences"]).rename({
                "id": "candidate_id"
            })
            if (
                not tables["summary.parquet"]
                .select("candidate_id", "sequence")
                .sort("candidate_id")
                .equals(
                    expected.select("candidate_id", "sequence").sort("candidate_id")
                )
            ):
                raise ValueError("Scorer changed candidate identity or sequence")
            for name, frame in tables.items():
                if (
                    name != "summary.parquet"
                    and frame
                    .select(pl.col("seq_id").alias("candidate_id"))
                    .join(expected, on="candidate_id", how="anti")
                    .height
                ):
                    raise ValueError("Detailed scores reference a foreign candidate")
        return result

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish raw evidence and rank only candidates with both finite scores."""
        candidates, generation, mutations = (
            pl.read_parquet(context.resolve_artifact(context.single_input(name)))
            for name in ("candidates", "generation", "mutations")
        )
        scores, details = {}, {}
        for key, result in results.items():
            if key in {"baseline", "annotation"}:
                continue
            model = key.split("-", 1)[0]
            for name, frame in score_tables(context, result).items():
                if name == "summary":
                    scores.setdefault(model, []).append(frame)
                else:
                    frame = frame.rename({"seq_id": "candidate_id"}).join(
                        candidates.select(KEY),
                        on="candidate_id",
                        how="left",
                        validate="m:1",
                    )
                    details.setdefault(f"{model}_{name}", []).append(frame)
        for key, error in errors.items():
            if key == "annotation":
                continue
            model, index = key.split("-", 1)
            scores.setdefault(model, []).append(
                candidates.filter(
                    pl.col("parent_id") == self.parents[int(index)].id
                ).select(
                    "candidate_id",
                    pl.lit(None, dtype=pl.Float64).alias("score"),
                    pl.lit(error).alias("error"),
                )
            )
        selection = rank_panel(
            add_scores(
                candidates,
                mutations,
                {
                    model: pl.concat([
                        frame.select("candidate_id", "score", "error")
                        for frame in frames
                    ])
                    for model, frames in scores.items()
                },
            ),
            mutations,
        )
        if "annotation" in results:
            annotated = score_tables(context, results["annotation"])
            annotations, germlines = annotated["annotations"], annotated["germlines"]
        else:
            annotations = candidates.select(
                *KEY,
                pl.lit(None, dtype=pl.Float64).alias("vh_pI"),
                pl.lit(None, dtype=pl.String).alias("vh_v_gene"),
                pl.lit(None, dtype=pl.String).alias("vh_j_gene"),
                pl.lit(errors["annotation"]).alias("annotation_error"),
            )
            germlines = pl.DataFrame(schema=GERMLINE_TABLE_SCHEMA)
        selection = selection.join(
            annotations, on=KEY, how="left", validate="1:1", maintain_order="left"
        )
        incomplete = (
            bool(errors)
            or generation["error"].is_not_null().any()
            or (not selection["evaluation_complete"].all())
            or selection["annotation_error"].is_not_null().any()
        )
        bundle = export_results(
            context,
            selection,
            generation,
            mutations,
            germlines,
            {
                name: pl.concat(frames, how="diagonal_relaxed")
                for name, frames in details.items()
            },
            self.parents,
            self.settings,
            SCIENTIFIC_VERSIONS,
            incomplete=incomplete,
        )
        return AppRunResult(
            status=AppRunStatus.PARTIAL if errors else AppRunStatus.SUCCEEDED,
            outputs=[bundle, json_output("incomplete", incomplete)],
        )


@dataclass
class NanobodyPublishNode(TaskProviderNode):
    """Reflect row-level failures in the shared lifecycle without discarding evidence."""

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        """Two local checks; this boundary never submits provider work."""
        return (
            TaskDefinition(
                "result", context.single_input("result").model_dump(mode="json")
            ),
            TaskDefinition(
                "complete", orjson.loads(context.read_input_bytes("incomplete"))
            ),
        )

    def recover_remote_task_result(
        self, context: NodeRunContext, task: TaskDefinition, expected_fingerprint: str
    ) -> AppRunResult | None:
        """Retain the complete file manifest for terminal-first successor recovery."""
        if task.task_key == "result":
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    republish_execution_artifact(
                        context.single_input("result")
                    ).model_copy(update={"name": "nanobody_results"})
                ],
            )
        if not task.scientific_payload:
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED, outputs=[json_output("complete", True)]
            )
        return None

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        """Conclusive local failure, not a hidden remote retry or dummy operation."""
        raise ValueError(
            "Some generation, evaluation or annotation results are unavailable; see result evidence"
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Let kernel-owned Task outcomes determine Completed versus Partial."""
        return AppRunResult(
            status=AppRunStatus.PARTIAL if errors else AppRunStatus.SUCCEEDED
        )


def build_nanobody_graph(
    parents: tuple[PreparedVH, ...], settings: NanobodySettings
) -> ExecutionGraph:
    """One scheduler, independent generation arms, one union and dual scoring."""
    if not 1 <= len(parents) <= 200 or len({parent.id for parent in parents}) != len(
        parents
    ):
        raise ValueError("Expected 1–200 unique prepared parents")
    if any(parent.preparation_version != PREPARATION_VERSION for parent in parents):
        raise ValueError("Prepared parents use an unsupported preparation version")
    graph = ExecutionGraph(
        "nanobody_humanization", scientific_versions=SCIENTIFIC_VERSIONS
    )
    generators = {
        method: graph.add_node(
            NanobodyGenerateNode(parents, settings, method),
            id=f"generate_{method}",
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
        )
        for method in METHODS
    }
    union = graph.add_node(
        NanobodyUnionNode(parents),
        id="union",
        inputs={
            method: ArtifactSelector(
                producing_node_id=node.node_id, pattern="generation.parquet"
            )
            for method, node in generators.items()
        },
        accept_partial_from=list(generators.values()),
        reuse_predecessor_publication=False,
    )
    evaluation = graph.add_node(
        NanobodyEvaluateNode(parents, settings),
        id="evaluate",
        inputs={
            name: ArtifactSelector(
                producing_node_id=union.node_id, pattern=f"{name}.parquet"
            )
            for name in ("candidates", "generation", "mutations")
        },
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    graph.add_node(
        NanobodyPublishNode(),
        id="publish",
        inputs={
            "result": ArtifactSelector(
                producing_node_id=evaluation.node_id, kind=ArtifactKind.DIRECTORY
            ),
            "incomplete": ArtifactSelector(
                producing_node_id=evaluation.node_id, pattern="incomplete.json"
            ),
        },
        accept_partial_from=[evaluation],
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
    """Thin Modal boundary over the same staged lifecycle used by other Tools."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()
    development: bool = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh the staged request before initializing the single writer."""
        initialize_execution_coordinator_host(self)
        execution_coordinator_identity(self)
        OUT_VOLUME.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionOverview:
        """Drive a root Run; never submit a second coordinator for an app call."""
        return self._adapter(development=development).run()

    @modal.method()
    def resume(self) -> ExecutionOverview:
        """Reconcile the existing Run without retrying conclusive failures."""
        return self._adapter().resume()

    @modal.method()
    def status(self) -> ExecutionOverview:
        """Read the shared durable execution projection."""
        return self._adapter().status()

    @modal.method()
    def result(self) -> AppRunResult:
        """Return terminal result references, not large scientific payloads."""
        return self._adapter().result()

    @modal.method()
    def cancel(self) -> ExecutionOverview:
        """Cancel only this Run's owned provider calls."""
        return self._adapter().cancel()

    @modal.method()
    def provider_calls(
        self,
        node_key: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        newest_first: bool = False,
    ) -> ProviderCallPage:
        """Page call identities for shared timing and log displays."""
        return self._adapter().provider_calls(
            node_key=node_key,
            cursor=UUID(cursor) if cursor else None,
            limit=limit,
            newest_first=newest_first,
        )

    @modal.method()
    def provider_call(self, provider_call_id: str) -> ProviderCallDiagnostic | None:
        """Read one exact owned call."""
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
        """Prepare explicit compatible successor recovery using shared ownership rules."""
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
        """Drive a prepared successor."""
        return self._adapter().drive_prepared()

    @modal.method()
    def restart_from(self, predecessor_execution_run_id: str) -> ExecutionOverview:
        """Verify the staged CLI request matches the predecessor scientific plan."""
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
        """Checkpoint best-effort without cancelling children on preemption."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()

    def _adapter(
        self, *, development: bool | None = None
    ) -> NanobodyExecutionCoordinator:
        from biomodals.app.design.abnativ2_vhh import app as abnativ_app
        from biomodals.app.design.hudiff_nb import app as hudiff_app

        execution_run_id, deployment = execution_coordinator_identity(self)
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected: NanobodyExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=OUT_VOLUME_MOUNTPOINT,
                output_volume=OUT_VOLUME,
                output_volume_name=OUT_VOLUME_NAME,
                provider_driver=development_modal_call_driver(
                    {
                        "abnativ2_vhh_humanize": abnativ_app.abnativ2_vhh_humanize,
                        "abnativ2_vhh_score": abnativ_app.abnativ2_vhh_score,
                        "hudiff_nb_humanize": hudiff_app.hudiff_nb_humanize,
                        "annotate_nanobody_candidates": annotate_nanobody_candidates,
                    },
                    workload_name=CONF.name,
                )
                if selected
                else ModalCallDriver(),
            ),
        )


@app.local_entrypoint()
def submit_nanobody_humanization_workflow(
    input_csv: str,
    run_id: str | None = None,
    root_seed: int = 0,
    hudiff_nb_candidate_count: int = 10,
    abnativ2_residue_score_threshold: float = 0.98,
    abnativ2_rasa_threshold: float = 0.15,
    abnativ2_max_relative_vhh_score_decrease: float = 0.05,
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
    """Prepare single VH domains, generate concurrently and score their exact union.

    Args:
        input_csv: CSV containing exactly id,vhh; 1–200 original constructs.
        run_id: Logical run label; defaults to the input filename stem.
        root_seed: Explicit HuDiff-Nb root seed, unsigned 32-bit integer.
        hudiff_nb_candidate_count: Native attempts per parent, 1–25; no refill.
        abnativ2_residue_score_threshold: Enhanced humanization residue threshold.
        abnativ2_rasa_threshold: Native relative solvent-accessibility threshold.
        abnativ2_max_relative_vhh_score_decrease: Allowed per-step VHH nativeness loss.
        wait: Wait for completion and print final Volume paths.
        max_containers: Global active provider-call ceiling.
        max_gpu_containers: Global GPU subset of the provider-call ceiling.
        dry_run: Validate local preparation and display the DAG without cloud calls.
        use_deployed_coordinator: Use the exact deployed workflow version.
        deployment_environment: Modal environment for the containing workflow.
        deployment_name: Containing deployment name, default NanobodyHumanizationWorkflow.
        deployment_version: Exact numeric deployment version.
        restart_from: Explicit compatible predecessor Run UUID.
    """
    values = locals()
    settings = NanobodySettings.model_validate({
        name: values[name] for name in NanobodySettings.model_fields
    })
    source = Path(input_csv).expanduser().resolve()
    content = read_bounded_file_bytes(
        source, field_name="Nanobody CSV", max_bytes=10 * 1024 * 1024
    )
    frame = pl.read_csv(BytesIO(content), infer_schema=False)
    if frame.columns != ["id", "vhh"] or not 1 <= frame.height <= 200:
        raise ValueError("Expected id,vhh CSV with 1–200 rows")
    parents, issues = prepare_batch([
        VHInput.model_validate(row) for row in frame.iter_rows(named=True)
    ])
    if issues:
        raise ValueError(
            "; ".join(f"Row {issue.row_index + 1}: {issue.message}" for issue in issues)
        )
    graph = build_nanobody_graph(parents, settings)
    total, gpu = resolve_provider_call_limits(
        max_containers=max_containers,
        max_gpu_containers=max_gpu_containers,
        default_max_containers=8,
        default_max_gpu_containers=2,
    )
    if dry_run:
        print_workflow_dag(graph.validate())
        return
    predecessor = UUID(restart_from) if restart_from else None
    if predecessor is not None and not use_deployed_coordinator:
        raise ValueError("Restart requires an exact deployed workflow version")
    deployment = DeploymentIdentity(
        deployment_environment if use_deployed_coordinator else "development",
        deployment_name or CONF.name,
        deployment_version if use_deployed_coordinator else 1,
    )
    # Environment preparation is CPU-only and precedes all GPU admissions.
    from biomodals.app.design.abnativ2_vhh.app import stage_abnativ2_vhh_models
    from biomodals.app.design.hudiff_nb.app import stage_hudiff_nb_models

    for name, local in (
        ("stage_abnativ2_vhh_models", stage_abnativ2_vhh_models),
        ("stage_hudiff_nb_models", stage_hudiff_nb_models),
    ):
        function = (
            modal.Function.from_name(
                deployment.deployment_name,
                name,
                environment_name=deployment.environment,
                version=deployment.deployment_version,
            )
            if use_deployed_coordinator
            else local
        )
        function.remote()
    execution_run_id = uuid4()
    stage_execution_request(
        OUT_VOLUME,
        execution_run_id,
        NanobodyExecutionRequest(
            run_name=sanitize_filename(run_id or source.stem),
            parents=parents,
            settings=settings,
            max_active_provider_calls=total,
            max_active_gpu_provider_calls=gpu,
        ),
    )
    stage_execution_launch(OUT_VOLUME, execution_run_id, predecessor)
    coordinator = execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=cast("Any", ExecutionCoordinator),
    )
    print(
        f"Deployment Identity: {deployment.environment}/{deployment.deployment_name}/v{deployment.deployment_version}\n"
        f"Execution Run ID: {execution_run_id}",
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
        print(f"Nanobody humanization: {overview.run.status.value}", flush=True)
        if overview.run.status.value in {"succeeded", "partial"}:
            result = AppRunResult.model_validate(coordinator.result.remote())
            for output in result.outputs:
                if output.name == "nanobody_results" and isinstance(
                    output.storage, VolumePath
                ):
                    print(
                        f"Selection table: volume={output.storage.volume_name} path={output.storage.path}/selection.csv",
                        flush=True,
                    )
