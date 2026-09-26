"""Real local graph, publication and fitting boundaries with deterministic providers."""

from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import orjson
import polars as pl
import pytest

from biomodals.app.design.mutation_ridge.runtime import run_ridge
from biomodals.app.misc.tabpfn.runtime import run_tabpfn
from biomodals.execution import DeploymentIdentity, RunStatus
from biomodals.execution.modal import (
    ProviderCallObservation,
    ProviderCallObservationKind,
)
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.workflow.protein_optimization import workflow
from biomodals.workflow.protein_optimization.design import OptimizationDesign
from biomodals.workflow.protein_optimization.execution import (
    OptimizationExecutionCoordinator,
    OptimizationExecutionRequest,
    load_execution_request,
    persist_execution_request,
)
from biomodals.workflow.protein_optimization.feature_stage import feature_tables
from biomodals.workflow.protein_optimization.nodes import optimization_graph
from biomodals.workflow.protein_optimization.publication import OptimizationManifest
from biomodals.workflow.protein_optimization.settings import (
    OptimizationSettings,
    PositionChoices,
)

CSV = 'mutations,label\n,0\nA:A1V,1\nA:A1V,3\nA:A1L,4\nB:C1V,3\n"A:A1V,B:C1V",5\n'


def _design(mode):
    return OptimizationDesign(
        measurements_csv=CSV,
        parental_fasta=">A\nAA\n>B\nCC\n",
        settings=OptimizationSettings(mode=mode, candidate_budget=10, seed=17),
    )


class Volume:
    """No network; local storage survives coordinator reopen for lifecycle tests."""

    def reload(self):
        """Local writes are already visible."""

    def commit(self):
        """Nothing to synchronize outside this process."""


@pytest.mark.parametrize("mode", ["combination", "exploration"])
def test_native_graph_outputs_and_reopen_without_repeat_work(
    tmp_path, monkeypatch, mode
):
    """Exercise both modes through the actual shared lifecycle and CSV terminal."""
    roots = {}
    for module, name in (
        (workflow.ridge_app, "ridge"),
        (workflow.tabpfn_app, "tabpfn"),
    ):
        root = tmp_path / name
        roots[name] = root
        monkeypatch.setattr(
            module,
            "CONF",
            SimpleNamespace(
                output_volume=Volume(),
                output_volume_name=name,
                output_volume_mountpoint=str(root),
            ),
        )
    fit_sizes = []

    class Estimator:
        def fit(self, x, y):
            fit_sizes.append(len(y))
            self.score = float(np.mean(y))

        def predict(self, x, **kwargs):
            return np.repeat(self.score, len(x))

    def encoder(sequences):
        return np.stack([
            np.random
            .default_rng(sum(map(ord, sequence)))
            .normal(size=1152)
            .astype(np.float32)
            for sequence in sequences
        ])

    monkeypatch.setattr(
        "biomodals.workflow.protein_optimization.feature_stage.native_encoder",
        lambda root: encoder,
    )
    monkeypatch.setattr(
        "biomodals.app.misc.tabpfn.runtime.native_regressor",
        lambda *args, **kwargs: Estimator(),
    )
    monkeypatch.setattr(
        "biomodals.app.misc.tabpfn.runtime.checkpoint", lambda root: root
    )
    design = _design(mode)
    request = OptimizationExecutionRequest(run_name="test", design=design)
    run_id = uuid4()
    workflow_root = tmp_path / "workflow"
    persist_execution_request(workflow_root, run_id, request)
    calls = []

    class Driver:
        def resolve(self, binding):
            return binding.function_name

        def spawn(self, operation, *, args, kwargs):
            calls.append((operation, kwargs))
            return str(len(calls) - 1)

        def observe(self, handle):
            operation, kwargs = calls[int(handle)]
            if operation == "mutation_ridge_score":
                result = run_ridge(
                    **kwargs, volume_root=roots["ridge"], volume_name="ridge"
                )
            elif operation == "prepare_protein_optimization_models":
                result = AppRunResult(
                    status=AppRunStatus.SUCCEEDED,
                    outputs=[
                        AppOutput(
                            name="prepared",
                            kind=ArtifactKind.REPORT,
                            storage=InlineBytes(
                                data=b"offline", filename="prepared.txt"
                            ),
                        )
                    ],
                )
            elif operation == "extract_protein_optimization_features":
                result = feature_tables(
                    **kwargs,
                    volume_root=roots["tabpfn"],
                    volume_name="tabpfn",
                    model_root=tmp_path,
                )
            elif operation == "tabpfn_fit_predict":
                result = run_tabpfn(
                    **kwargs,
                    volume_root=roots["tabpfn"],
                    volume_name="tabpfn",
                    model_root=tmp_path,
                )
            else:
                raise AssertionError(operation)
            return ProviderCallObservation(
                ProviderCallObservationKind.SUCCEEDED, result=result
            )

        def cancel(self, handle):
            pass

    def host():
        return OptimizationExecutionCoordinator(
            execution_run_id=run_id,
            deployment=DeploymentIdentity("main", "ProteinOptimizationWorkflow", 1),
            volume_root=workflow_root,
            output_volume=Volume(),
            output_volume_name="workflow",
            provider_driver=Driver(),
            external_checker=workflow._check_external,
            poll_interval_seconds=0,
        )

    coordinator = host()
    try:
        overview = coordinator.run()
        assert overview.run.status == RunStatus.SUCCEEDED, overview
        result = coordinator.result()
        root = workflow_root / result.outputs[0].storage.path
        manifest = OptimizationManifest.model_validate_json(
            (root / "manifest.json").read_bytes()
        )
        candidates = pl.read_csv(root / "candidates.csv")
        assert candidates.height == design.candidate_count(design.dataset())
        assert candidates["id"].n_unique() == candidates.height
        assert candidates["predicted_label"].is_finite().all()
        assert not set(candidates["mutations"].to_list()) & set(
            design.dataset().measurements["mutations"].to_list()
        )
        assert manifest.validation.training_variants == 5
        if mode == "combination":
            assert [op for op, _ in calls] == ["mutation_ridge_score"]
            assert candidates["sequence_A"].to_list() == ["LA"]
        else:
            assert [op for op, _ in calls] == [
                "prepare_protein_optimization_models",
                "extract_protein_optimization_features",
                "tabpfn_fit_predict",
            ]
            assert candidates["n_new_mutations"].min() >= 1
            assert fit_sizes[-1] == 5
            assert manifest.validation.evaluated_variants > 0
            assert manifest.fitted_feature_widths[-1] == 4
        expected_calls = len(calls)
    finally:
        coordinator.close()
    reopened = host()
    try:
        assert reopened.run().run.status == RunStatus.SUCCEEDED
        assert len(calls) == expected_calls
        assert reopened.result() == result
    finally:
        reopened.close()


def test_request_pins_scientific_versions_not_resource_limits(tmp_path):
    """Retained inputs remain readable; incompatible compute cannot resume silently."""
    request = OptimizationExecutionRequest(
        run_name="label", design=_design("exploration")
    )
    run_id = uuid4()
    persist_execution_request(tmp_path, run_id, request)
    assert load_execution_request(tmp_path, run_id) == request
    assert (
        replace(
            request, max_active_provider_calls=100, max_active_gpu_provider_calls=40
        ).execution_plan
        == request.execution_plan
    )
    saved = orjson.loads(request.to_bytes())
    saved["scientific_versions"]["esmc"] = "older"
    restored = OptimizationExecutionRequest.from_bytes(orjson.dumps(saved))
    assert restored.design == request.design
    with pytest.raises(ValueError, match="scientific versions"):
        _ = restored.execution_plan
    with pytest.raises(ValueError, match="scientific versions"):
        _ = replace(request, scientific_versions={}).execution_plan


def test_empty_exploration_and_combination_graphs_have_no_model_preparation():
    """An empty design space avoids pointless GPU initialization and fitting."""
    design = _design("exploration")
    frozen = design.model_copy(
        update={
            "settings": design.settings.model_copy(
                update={
                    "positions": (
                        PositionChoices(chain_id="A", position=1, amino_acids=""),
                    )
                }
            )
        }
    )
    graph = optimization_graph(frozen).validate()
    assert list(graph.nodes) == ["publish"]
    combination = optimization_graph(_design("combination")).validate()
    assert list(combination.nodes) == ["fit_score_combinations", "publish"]


def test_workflow_cli_dry_run_stages_no_files_or_models(tmp_path, monkeypatch, capsys):
    """Both modes can be validated and inspected without paid calls."""
    csv, fasta = tmp_path / "measurements.csv", tmp_path / "parent.fasta"
    csv.write_text(CSV)
    fasta.write_text(">A\nAA\n>B\nCC\n")
    monkeypatch.setattr(
        workflow,
        "execution_coordinator_handle",
        lambda **kwargs: pytest.fail("Cloud access in dry run"),
    )
    for mode in ("combination", "exploration"):
        workflow.submit_protein_optimization_workflow(
            input_csv=str(csv), parental_fasta=str(fasta), mode=mode, dry_run=True
        )
        assert "publish" in capsys.readouterr().out
    assert workflow.CONF.depends_on_apps == ("mutation_ridge", "tabpfn")
    operations = workflow.app._local_state.functions
    assert {
        "mutation_ridge_score",
        "tabpfn_fit_predict",
        "extract_protein_optimization_features",
    } <= set(operations)
