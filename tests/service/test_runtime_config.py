"""Tool runtime configuration contracts."""

# ruff: noqa: D103

from pathlib import Path

import pytest

from biomodals.service.config import ServiceSettings
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import ServiceStore
from biomodals.service.tools import TOOLS


def _configuration(tmp_path: Path) -> RuntimeConfiguration:
    settings = ServiceSettings.from_environment({
        "BIOMODALS_STATE_DIR": str(tmp_path / "state"),
        "BIOMODALS_CACHE_DIR": str(tmp_path / "cache"),
    })
    store = ServiceStore(settings.database_path)
    store.initialize()
    return RuntimeConfiguration(store, settings, tool_definitions=TOOLS)


def test_registered_tool_defaults_are_explicit(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)

    assert configuration.tool_names() == ("gromacs", "alphafold3")
    assert configuration.tool("gromacs").max_active_provider_calls.value == 3
    assert configuration.tool("alphafold3").max_active_provider_calls.value == 4


def test_container_limits_are_atomic_and_gpu_fits_total(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    configuration.set_tool(
        "alphafold3",
        max_active_provider_calls=12,
        max_active_gpu_provider_calls=3,
    )

    effective = configuration.tool("alphafold3")
    assert effective.max_active_provider_calls.value == 12
    assert effective.max_active_gpu_provider_calls.value == 3
    with pytest.raises(ValueError, match="GPU containers"):
        configuration.set_tool("alphafold3", max_active_provider_calls=2)
