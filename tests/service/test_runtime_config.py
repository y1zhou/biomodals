"""Database-backed runtime configuration precedence."""

# ruff: noqa: D103

from pathlib import Path
from uuid import uuid4

import pytest

from biomodals.service.config import ServiceSettings
from biomodals.service.runtime_config import RuntimeConfiguration, SettingOverrideError
from biomodals.service.store import ServiceStore


def _configuration(
    tmp_path: Path,
    *,
    process: dict[str, str] | None = None,
    config_file: str = "",
) -> RuntimeConfiguration:
    environment = dict(process or {})
    if config_file:
        path = tmp_path / "service.env"
        path.write_text(config_file)
        path.chmod(0o600)
        environment["BIOMODALS_API_CONF_ENV"] = str(path)
    settings = ServiceSettings.from_environment(environment)
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    return RuntimeConfiguration(store, settings)


def test_database_overrides_file_and_updates_are_live(tmp_path: Path) -> None:
    configuration = _configuration(
        tmp_path,
        config_file=(
            "BIOMODALS_MODAL_ENVIRONMENT=file-env\n"
            "BIOMODALS_GROMACS_APP=FileApp\n"
            "BIOMODALS_GROMACS_APP_VERSION=3\n"
            "BIOMODALS_GROMACS_ACTIVE_LIMIT=3\n"
            "BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT=8\n"
        ),
    )

    assert configuration.modal_environment().value == "file-env"
    assert configuration.workload("gromacs").modal_app_name.value == "FileApp"
    assert configuration.workload("gromacs").modal_app_version.value == 3

    configuration.update_environment(
        modal_environment="database-env",
        global_active_job_limit=12,
    )
    configuration.set_workload(
        "gromacs",
        modal_app_name="DatabaseApp",
        modal_app_version=7,
        active_job_limit=5,
    )

    assert configuration.modal_environment().value == "database-env"
    assert configuration.modal_environment().source == "database"
    assert configuration.global_active_job_limit().value == 12
    workload = configuration.workload("gromacs")
    assert workload.modal_app_name.value == "DatabaseApp"
    assert workload.modal_app_version.value == 7
    assert workload.active_job_limit.value == 5


def test_reset_removes_only_one_database_override(tmp_path: Path) -> None:
    configuration = _configuration(
        tmp_path,
        config_file=(
            "BIOMODALS_MODAL_ENVIRONMENT=file-env\n"
            "BIOMODALS_GROMACS_APP=FileApp\n"
            "BIOMODALS_GROMACS_APP_VERSION=3\n"
            "BIOMODALS_GROMACS_ACTIVE_LIMIT=3\n"
            "BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT=8\n"
        ),
    )
    configuration.update_environment(
        modal_environment="database-env",
        global_active_job_limit=12,
    )
    configuration.set_workload(
        "gromacs",
        modal_app_name="DatabaseApp",
        modal_app_version=7,
        active_job_limit=5,
    )

    configuration.update_environment(global_active_job_limit=None)
    configuration.set_workload("gromacs", active_job_limit=None)

    modal_environment = configuration.modal_environment()
    assert modal_environment.value == "database-env"
    assert modal_environment.source == "database"
    assert configuration.global_active_job_limit().value == 8
    assert configuration.global_active_job_limit().source == "configuration_file"
    workload = configuration.workload("gromacs")
    assert workload.modal_app_name.value == "DatabaseApp"
    assert workload.modal_app_name.source == "database"
    assert workload.modal_app_version.value == 7
    assert workload.modal_app_version.source == "database"
    assert workload.active_job_limit.value == 3
    assert workload.active_job_limit.source == "configuration_file"


def test_process_environment_is_effective_and_read_only(tmp_path: Path) -> None:
    configuration = _configuration(
        tmp_path,
        process={"BIOMODALS_MODAL_ENVIRONMENT": "process-env"},
        config_file="BIOMODALS_MODAL_ENVIRONMENT=file-env\n",
    )

    setting = configuration.modal_environment()

    assert setting.value == "process-env"
    assert setting.source == "process_environment"
    assert setting.editable is False
    with pytest.raises(SettingOverrideError, match="environment variable"):
        configuration.update_environment(modal_environment="database-env")


def test_admission_resolves_database_settings_inside_store_transaction(
    tmp_path: Path,
) -> None:
    configuration = _configuration(
        tmp_path,
        config_file=(
            "BIOMODALS_MODAL_ENVIRONMENT=file-env\n"
            "BIOMODALS_GROMACS_APP=FileApp\n"
            "BIOMODALS_GROMACS_APP_VERSION=3\n"
            "BIOMODALS_GROMACS_ACTIVE_LIMIT=3\n"
            "BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT=8\n"
        ),
    )
    user = configuration.store.create_user(
        email="admin@example.com",
        display_name="Admin",
        token_digest=b"token",
        token_expires_at=100,
        now=1,
        is_admin=True,
    )
    activated = configuration.store.set_password_from_token(
        b"token",
        password_hash="test-hash",  # noqa: S106 - test-only hash
        session_token_digest=b"session",
        csrf_digest=b"csrf",
        now=1,
        absolute_expires_at=100,
    )
    assert activated is not None
    admission_configuration = configuration.admission_configuration("gromacs")

    configuration.update_environment(
        modal_environment="database-env",
        global_active_job_limit=12,
    )
    configuration.set_workload(
        "gromacs",
        modal_app_name="DatabaseApp",
        modal_app_version=7,
        active_job_limit=5,
    )
    job = configuration.store.admit_job(
        owner_user_id=user.user_id,
        display_name="Simulation",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        parameters_json="{}",
        configuration=admission_configuration,
        now=2,
    ).job

    assert job.modal_environment == "database-env"
    assert job.modal_app_name == "DatabaseApp"
    assert job.modal_app_version == 7
