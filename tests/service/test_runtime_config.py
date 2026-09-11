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

    assert configuration.tool_names() == ("gromacs", "alphafold3", "humanization")
    assert configuration.tool("gromacs").max_active_provider_calls == 16
    assert configuration.tool("gromacs").max_active_gpu_provider_calls == 2
    assert configuration.tool("alphafold3").max_active_provider_calls == 16
    assert configuration.tool("alphafold3").max_active_gpu_provider_calls == 2
    assert configuration.tool("humanization").max_active_provider_calls == 16
    assert configuration.tool("humanization").max_active_gpu_provider_calls == 10


@pytest.mark.parametrize(
    "tool,gpu_limit", [("gromacs", 3), ("alphafold3", 3), ("humanization", 15)]
)
def test_container_limits_follow_active_job_limit(
    tmp_path: Path, tool, gpu_limit
) -> None:
    configuration = _configuration(tmp_path)
    configuration.set_tool(tool, active_job_limit=3)

    effective = configuration.tool(tool)
    assert effective.max_active_provider_calls == 24
    assert effective.max_active_gpu_provider_calls == gpu_limit


@pytest.mark.parametrize("tool", [tool.key for tool in TOOLS])
def test_disabled_tool_keeps_positive_provider_limit_floors(tmp_path, tool):
    configuration = _configuration(tmp_path)
    configuration.set_tool(tool, active_job_limit=0)

    effective = configuration.tool(tool)

    assert effective.active_job_limit.value == 0
    assert effective.max_active_provider_calls == 1
    assert effective.max_active_gpu_provider_calls == 1


def test_modal_app_name_comes_from_startup_configuration(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)

    effective = configuration.tool("gromacs")
    assert effective.modal_app_name == "Gromacs"
