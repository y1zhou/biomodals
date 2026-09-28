"""One execution graph, selected scientific mode, and a complete CSV terminal."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from biomodals.app.design.mutation_ridge.execution import MutationRidgeNode
from biomodals.app.misc.tabpfn.execution import TabPFNNode, TabPFNRequest
from biomodals.execution import ExecutionGraph, ExecutionPlanMetadata
from biomodals.execution.nodes import (
    CoordinatorNode,
    NodeRunContext,
    ProviderCallSpec,
    ProviderNode,
)
from biomodals.schema import AppRunResult, ArtifactKind, ArtifactSelector
from biomodals.workflow.protein_optimization.design import OptimizationDesign
from biomodals.workflow.protein_optimization.publication import publish_candidates
from biomodals.workflow.protein_optimization.validation import exploration_folds

PREPARE_OPERATION = "prepare_protein_optimization_models"
FEATURE_OPERATION = "extract_protein_optimization_features"


@dataclass
class PrepareModelsNode(ProviderNode):
    """One tracked, CPU-only readiness stage before either scientific GPU task."""

    scientific_versions: dict[str, str]

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """The kernel owns this provisioning call just like inference calls."""
        return ProviderCallSpec(
            function_name=PREPARE_OPERATION,
            uses_gpu=False,
            kwargs={"scientific_versions": self.scientific_versions},
        )


@dataclass
class ProteinFeaturesNode(ProviderNode):
    """A bounded candidate pool plus all training variants share frozen features."""

    design: OptimizationDesign

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """A stable run-owned key supports redelivery without cross-Job caches."""
        return ProviderCallSpec(
            function_name=FEATURE_OPERATION,
            uses_gpu=True,
            runtime_image_key="protein-esmc600m",
            kwargs={
                "design_json": self.design.model_dump_json(),
                "scientific_versions": self.design.scientific_versions(),
                "output_key": sha256(
                    f"{context.execution_run_id}/{context.node_id}/{context.task_key}".encode()
                ).hexdigest(),
            },
        )


@dataclass
class ProteinTabPFNNode(ProviderNode):
    """Delegate fit/evaluate/predict to the same app operation used by table CLI."""

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """The feature publication carries only typed paths, schema and holdouts."""
        request = TabPFNRequest.model_validate(
            context.single_input("features").metadata["tabpfn_request"]
        )
        return TabPFNNode(request).prepare_remote(context)


@dataclass
class PublishOptimizationNode(CoordinatorNode):
    """Local bounded consolidation, not another scientific provider call."""

    design: OptimizationDesign

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Refresh external producer volumes before exact-identity consolidation."""
        from biomodals.app.design.mutation_ridge import app as ridge_app
        from biomodals.app.misc.tabpfn import app as tabpfn_app

        roots = {}
        required = {
            artifact.storage.volume_name
            for artifacts in context.inputs.values()
            for artifact in artifacts
        }
        for conf in (ridge_app.CONF, tabpfn_app.CONF):
            if conf.output_volume_name in required:
                conf.output_volume.reload()
                roots[conf.output_volume_name] = Path(conf.output_volume_mountpoint)
        return publish_candidates(
            self.design,
            context,
            roots=roots,
            scored=context.single_input("scored")
            if "scored" in context.inputs
            else None,
            features=context.single_input("features")
            if "features" in context.inputs
            else None,
            validation=context.single_input("validation")
            if "validation" in context.inputs
            else None,
        )


def optimization_graph(design: OptimizationDesign) -> ExecutionGraph:
    """Never initialize embeddings or schedule GPUs for Combination."""
    graph = ExecutionGraph(
        "protein_optimization",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="protein_optimization",
            scientific_payload={"design_digest": design.digest()},
            scientific_versions=design.scientific_versions(),
        ),
    )
    if design.settings.mode == "combination":
        score = graph.add_node(
            MutationRidgeNode(design.ridge_request()), id="fit_score_combinations"
        )
        inputs = {
            "scored": ArtifactSelector(
                producing_node_id=score.node_id, pattern="candidates.csv"
            ),
            "validation": ArtifactSelector(
                producing_node_id=score.node_id, pattern="validation.json"
            ),
        }
    elif design.candidate_count(design.dataset()) == 0:
        inputs = {}
    else:
        prepare = graph.add_node(
            PrepareModelsNode(design.scientific_versions()),
            id="prepare_models",
            reuse_predecessor_publication=False,
        )
        features = graph.add_node(
            ProteinFeaturesNode(design), id="extract_features", depends_on=[prepare]
        )
        feature_selector = ArtifactSelector(
            producing_node_id=features.node_id, kind=ArtifactKind.DIRECTORY
        )
        score = graph.add_node(
            ProteinTabPFNNode(),
            id="fit_score_exploration",
            inputs={"features": feature_selector},
        )
        inputs = {
            "scored": ArtifactSelector(
                producing_node_id=score.node_id, pattern="predictions.csv"
            ),
            "features": feature_selector,
        }
        if exploration_folds(design.dataset().variants, seed=design.settings.seed):
            inputs["validation"] = ArtifactSelector(
                producing_node_id=score.node_id, pattern="validation.csv"
            )
    graph.add_node(PublishOptimizationNode(design), id="publish", inputs=inputs)
    return graph
