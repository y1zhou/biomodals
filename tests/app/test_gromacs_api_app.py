"""Composition contract for the Modal-hosted GROMACS API."""

# ruff: noqa: D101,D102,D103,D107

from biomodals.helper.catalog import get_catalog


def test_gromacs_api_is_discoverable() -> None:
    assert "gromacs_api" in get_catalog("app", use_absolute_paths=True)


def test_gromacs_api_composes_gromacs_app() -> None:
    from biomodals.app.service import gromacs_api_app

    assert "api" in gromacs_api_app.app.registered_functions
    assert "run_gromacs_job" in gromacs_api_app.app.registered_functions
