"""Tool runtime configuration contracts."""

# ruff: noqa: D103

from pathlib import Path

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


def test_container_limits_follow_active_job_limit(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    configuration.set_tool("alphafold3", active_job_limit=3)

    effective = configuration.tool("alphafold3")
    assert effective.max_active_provider_calls == 24
    assert effective.max_active_gpu_provider_calls == 3


def test_modal_app_name_comes_from_startup_configuration(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)

    effective = configuration.tool("gromacs")
    assert effective.modal_app_name == "Gromacs"
