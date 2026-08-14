"""Packaging contracts for execution-kernel Modal images."""

# ruff: noqa: D101,D103

import ast
import importlib.util
import sys
from enum import StrEnum
from pathlib import Path
from types import ModuleType

import pytest

from biomodals.app.bioinfo import gromacs_app, rosetta_app
from biomodals.app.design import boltzgen_app
from biomodals.app.fold import (
    abcfold2_app,
    alphafold3_app,
    protenix_app,
)
from biomodals.app.score import af3score_app, ensirna_app, oligoformer_app
from biomodals.workflow import ppiflow_workflow


def _source_modules(image) -> set[str]:
    original = next(
        (
            value
            for key, value in image.__dict__.items()
            if key.startswith("_sync_original")
        ),
        image,
    )
    return set(original._added_python_source_set)


@pytest.mark.parametrize(
    ("image", "required_modules"),
    (
        (
            gromacs_app.runtime_image,
            {
                "biomodals.app.bioinfo.gromacs_execution",
                "biomodals.app.bioinfo.gromacs_execution_runtime",
            },
        ),
        (
            gromacs_app.biotite_image,
            {
                "biomodals.app.bioinfo.gromacs_execution",
                "biomodals.app.bioinfo.gromacs_execution_runtime",
            },
        ),
        (
            rosetta_app.runtime_image,
            {"biomodals.app.bioinfo.rosetta"},
        ),
        (
            boltzgen_app.runtime_image,
            {"biomodals.app.design.boltzgen"},
        ),
        (
            abcfold2_app.runtime_image,
            {"biomodals.app.fold.abcfold2_execution"},
        ),
        (
            abcfold2_app.download_image,
            {"biomodals.app.fold.abcfold2_execution"},
        ),
        (
            alphafold3_app.runtime_image,
            {"biomodals.app.fold.alphafold3"},
        ),
        (
            alphafold3_app.sharding_image,
            {"biomodals.app.fold.alphafold3"},
        ),
        (
            protenix_app.runtime_image,
            {"biomodals.app.fold.protenix_execution"},
        ),
        (
            ensirna_app.runtime_image,
            {"biomodals.app.score.ensirna_execution"},
        ),
        (
            af3score_app.runtime_image,
            {
                "biomodals.app.score.af3score_execution",
                "biomodals.app.score.af3score_publications",
            },
        ),
        (
            oligoformer_app.runtime_image,
            {"biomodals.app.score.oligoformer_execution"},
        ),
        (
            ppiflow_workflow.ligandmpnn_task_image,
            {
                "biomodals.app.design.ligandmpnn_app",
                "biomodals.workflow.ppiflow",
            },
        ),
    ),
)
def test_execution_images_include_app_owned_modules(
    image,
    required_modules: set[str],
) -> None:
    assert required_modules <= _source_modules(image)


@pytest.mark.parametrize(
    ("module_name", "feature_version"),
    (
        ("biomodals.app.fold.protenix_execution", (3, 11)),
        ("biomodals.app.score.ensirna_execution", (3, 10)),
        ("biomodals.app.score.oligoformer_execution", (3, 10)),
        ("biomodals.workflow.ppiflow.staging", (3, 11)),
        ("biomodals.execution.pull_worker", (3, 11)),
        ("biomodals.app.fold.alphafold3.generation_claims", (3, 11)),
        ("biomodals.app.fold.alphafold3.inference_inputs", (3, 11)),
        ("biomodals.app.fold.alphafold3.search_pipeline", (3, 11)),
        ("biomodals.app.fold.alphafold3.seed_predictions", (3, 11)),
        ("biomodals.app.design.ligandmpnn_app", (3, 11)),
        ("biomodals.workflow.ppiflow.ligandmpnn_runtime", (3, 11)),
        ("biomodals.workflow.ppiflow.manifests", (3, 11)),
        ("biomodals.workflow.ppiflow.runtime_context", (3, 11)),
        ("biomodals.workflow.ppiflow.runtime_support", (3, 11)),
        ("biomodals.workflow.ppiflow.tables", (3, 11)),
    ),
)
def test_low_python_execution_sources_parse(
    module_name: str,
    feature_version: tuple[int, int],
) -> None:
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin is not None
    source_path = Path(spec.origin)
    ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
        feature_version=feature_version,
    )


def test_python310_shared_runtime_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backports = ModuleType("backports")
    backports.__dict__["__path__"] = []
    strenum = ModuleType("backports.strenum")
    strenum.__dict__["StrEnum"] = StrEnum
    backports.__dict__["strenum"] = strenum
    monkeypatch.setitem(sys.modules, "backports", backports)
    monkeypatch.setitem(sys.modules, "backports.strenum", strenum)
    monkeypatch.setattr(sys, "version_info", (3, 10))

    loaded = {}
    for module_name in (
        "biomodals.execution.model",
        "biomodals.execution.modal",
        "biomodals.helper.artifacts",
    ):
        source_spec = importlib.util.find_spec(module_name)
        assert source_spec is not None and source_spec.origin is not None
        test_name = f"_python310_{module_name.replace('.', '_')}"
        spec = importlib.util.spec_from_file_location(test_name, source_spec.origin)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, test_name, module)
        spec.loader.exec_module(module)
        loaded[module_name] = module

    assert loaded["biomodals.execution.model"].RunStatus.SUCCEEDED == "succeeded"
    assert (
        loaded["biomodals.execution.modal"].ModalCallObservationKind.SUCCEEDED
        == "succeeded"
    )
    assert loaded["biomodals.helper.artifacts"].utc_now().endswith("+00:00")
