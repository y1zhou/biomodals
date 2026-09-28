"""Offline staged table, native call boundary, and durable CSV publication checks."""

import sys
from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import polars as pl
import pytest

from biomodals.app.misc.tabpfn import app as tabpfn_app
from biomodals.app.misc.tabpfn.execution import (
    OPERATION,
    PREPARE_OPERATION,
    TableFile,
    TabPFNNode,
    TabPFNRequest,
    tabpfn_graph,
)
from biomodals.app.misc.tabpfn.models import (
    RUNTIME_IDENTITY,
    checkpoint,
    native_regressor,
)
from biomodals.app.misc.tabpfn.runtime import run_tabpfn
from biomodals.app.misc.tabpfn.tables import (
    MAX_TRAIN_ROWS,
    TableFeature,
    TableSchema,
    read_table_file,
)
from biomodals.execution.definition_plan import execution_plan
from biomodals.execution.nodes import NodeRunContext


def _request(root):
    directory = root / "inputs" / ("a" * 32)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name, data in (
        ("training", b"id,x,y\na,1,2\nb,2,4\n"),
        ("inference", b"id,x\nx,3\ny,4\n"),
    ):
        path = directory / f"{name}.csv"
        path.write_bytes(data)
        files.append(
            TableFile(
                path=path.relative_to(root).as_posix(),
                sha256=sha256(data).hexdigest(),
                size_bytes=len(data),
            )
        )
    return TabPFNRequest(
        training=files[0],
        inference=files[1],
        table_schema=TableSchema(
            target="y", identifier="id", features=(TableFeature(name="x"),)
        ),
        batch_size=1,
    )


def test_fit_predict_publication_redelivery_and_tampering(tmp_path, monkeypatch):
    """Same completed task never refits, and result bytes are content-bound."""
    request = _request(tmp_path)
    events = []
    monkeypatch.setattr(
        "biomodals.app.misc.tabpfn.runtime.checkpoint",
        lambda root: root / "verified.safetensors",
    )

    class Estimator:
        def fit(self, x, y):
            events.append(("fit", len(x)))
            np.testing.assert_array_equal(y, [2, 4])
            self.inferred_feature_schema_ = SimpleNamespace(
                features=[SimpleNamespace(modality=SimpleNamespace(value="numerical"))]
            )

        def predict(self, x, *, output_type):
            assert output_type == "mean"
            events.append(("predict", len(x)))
            return x[:, 0] * 2

    monkeypatch.setattr(
        "biomodals.app.misc.tabpfn.runtime.native_regressor",
        lambda *a, **k: Estimator(),
    )
    kwargs = dict(
        output_key="b" * 64,
        runtime_identity=RUNTIME_IDENTITY,
        volume_root=tmp_path,
        volume_name="test",
        model_root=tmp_path,
    )
    first = run_tabpfn(request.model_dump_json(), **kwargs)
    assert events == [("fit", 2), ("predict", 1), ("predict", 1)]
    output = first.outputs[0]
    csv = tmp_path / output.storage.path
    assert pl.read_csv(csv).to_dict(as_series=False) == {
        "id": ["x", "y"],
        "predicted_label": [6.0, 8.0],
    }
    assert output.metadata["feature_schema"] == request.table_schema.model_dump(
        mode="json"
    )
    assert output.metadata["fitted_feature_modalities"] == [["numerical"]]
    assert run_tabpfn(request.model_dump_json(), **kwargs) == first
    assert len(events) == 3
    with pytest.raises(ValueError, match="different request"):
        run_tabpfn(request.model_copy(update={"seed": 1}).model_dump_json(), **kwargs)
    csv.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        run_tabpfn(request.model_dump_json(), **kwargs)


def test_input_integrity_and_missing_weights_fail_before_fit(tmp_path, monkeypatch):
    """Neither user paths nor implicit weight downloads bypass preparation."""
    request = _request(tmp_path)
    with pytest.raises(ValueError):
        TableFile(path="../outside.csv", sha256="0" * 64, size_bytes=1)
    (tmp_path / request.training.path).write_bytes(b"corrupt")
    monkeypatch.setattr(
        "biomodals.app.misc.tabpfn.runtime.native_regressor",
        lambda *a, **k: pytest.fail("Unexpected fitting"),
    )
    with pytest.raises(ValueError, match="verification"):
        run_tabpfn(
            request.model_dump_json(),
            output_key="b" * 64,
            runtime_identity=RUNTIME_IDENTITY,
            volume_root=tmp_path,
            volume_name="test",
            model_root=tmp_path,
        )
    with pytest.raises(ValueError, match="prepare the environment"):
        checkpoint(tmp_path)


def test_parquet_and_csv_share_schema_boundary(tmp_path):
    """Workflow features preserve explicit column order and numeric semantics."""
    request = _request(tmp_path)
    csv = read_table_file(
        tmp_path / request.training.path, request.table_schema, training=True
    )
    path = tmp_path / "features.parquet"
    csv.select("y", "id", "x").write_parquet(path)
    assert read_table_file(path, request.table_schema, training=True).equals(csv)
    csv.head(1).write_parquet(path)
    with pytest.raises(ValueError, match="between 2"):
        read_table_file(path, request.table_schema, training=True)


def test_graph_models_are_prepared_before_gpu_and_staging_is_not_science(tmp_path):
    """Changing transport paths does not change scientific plan identity."""
    request = _request(tmp_path)
    graph = tabpfn_graph(request).validate()
    assert graph.dependencies["fit_predict"] == frozenset({"prepare_models"})
    context = NodeRunContext(
        uuid4(), "run", "fit_predict", "node", tmp_path, tmp_path, {}
    )
    call = TabPFNNode(request).prepare_remote(context)
    assert call.function_name == OPERATION and call.uses_gpu
    prepare = graph.nodes["prepare_models"].node.prepare_remote(context)
    assert prepare.function_name == PREPARE_OPERATION and not prepare.uses_gpu
    moved = request.model_copy(
        update={
            "training": request.training.model_copy(
                update={"path": "inputs/" + "f" * 32 + "/training.csv"}
            )
        }
    )
    original_plan = execution_plan(graph, workload_run_key="run")
    moved_plan = execution_plan(tabpfn_graph(moved).validate(), workload_run_key="run")
    assert original_plan == moved_plan


def test_cli_dry_run_checks_schema_without_cloud_calls(tmp_path, monkeypatch, capsys):
    """A dry run validates full native tables before any upload or provisioning."""
    request = _request(tmp_path)
    schema = tmp_path / "schema.json"
    schema.write_text(request.table_schema.model_dump_json())
    monkeypatch.setattr(
        tabpfn_app.orchestrator,
        "execution_coordinator_handle",
        lambda **k: pytest.fail("Unexpected Modal call"),
    )
    tabpfn_app.submit_tabpfn_task(
        training_csv=str(tmp_path / request.training.path),
        inference_csv=str(tmp_path / request.inference.path),
        schema_json=str(schema),
        dry_run=True,
    )
    assert "fit_predict" in capsys.readouterr().out


def test_worker_refreshes_both_volumes_before_native_read_and_commits(
    tmp_path, monkeypatch
):
    """Warm consumers see preparation/input commits, with inference model read-only."""
    events = []

    class Volume:
        def __init__(self, name):
            self.name = name

        def reload(self):
            events.append(self.name + ":reload")

        def commit(self):
            events.append(self.name + ":commit")

    class Config:
        output_volume = Volume("output")
        output_volume_name = "test"
        output_volume_mountpoint = str(tmp_path)

    def run(*args, **kwargs):
        events.append("native")
        return "result"

    monkeypatch.setattr(tabpfn_app, "CONF", Config())
    monkeypatch.setattr(tabpfn_app, "MODEL_VOLUME", Volume("model"))
    monkeypatch.setattr(tabpfn_app, "run_tabpfn", run)
    assert (
        tabpfn_app.tabpfn_fit_predict.get_raw_f()("{}", "a" * 64, RUNTIME_IDENTITY)
        == "result"
    )
    assert events == ["output:reload", "model:reload", "native", "output:commit"]
    events.clear()
    monkeypatch.setattr(
        tabpfn_app, "provision_checkpoint", lambda root: events.append("provision")
    )
    result = tabpfn_app.prepare_tabpfn_models.get_raw_f()(RUNTIME_IDENTITY)
    assert result.outputs[0].storage.data.decode() == RUNTIME_IDENTITY
    assert events == ["model:reload", "provision", "model:commit"]


def test_native_categorical_override_covers_every_admitted_training_row(
    monkeypatch, tmp_path
):
    """Numeric-looking categories above the native default remain categorical."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    monkeypatch.setitem(
        sys.modules, "tabpfn", SimpleNamespace(TabPFNRegressor=lambda **kwargs: kwargs)
    )
    estimator = native_regressor(
        tmp_path / "verified.safetensors", categorical_indices=[1]
    )
    assert estimator["categorical_features_indices"] == [1]
    assert (
        estimator["inference_config"]["MAX_UNIQUE_FOR_CATEGORICAL_FEATURES"]
        == MAX_TRAIN_ROWS
    )
