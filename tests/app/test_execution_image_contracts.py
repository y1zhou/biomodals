"""Packaging contracts for execution-kernel Modal images."""

# ruff: noqa: D101,D103

import ast
import importlib.util
from pathlib import Path

import pytest

from biomodals.app.bioinfo import gromacs_app, rosetta_app
from biomodals.app.design import boltzgen_app
from biomodals.app.fold import (
    abcfold2_app,
    alphafold3_app,
    protenix_app,
)
from biomodals.app.score import af3score_app, ensirna_app, oligoformer_app


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
