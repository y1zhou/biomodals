"""Live non-secret service configuration with explicit source precedence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from biomodals.service.config import ServiceSettings

if TYPE_CHECKING:
    from biomodals.service.store import ServiceStore

SettingSource = Literal[
    "process_environment",
    "database",
    "configuration_file",
    "default",
]


class SettingOverrideError(ValueError):
    """Raised when an administrator edits a process-controlled setting."""


class _Unchanged:
    """Sentinel distinguishing an omitted PATCH field from a null reset."""


_UNCHANGED = _Unchanged()


@dataclass(frozen=True, slots=True)
class EffectiveSetting[Value: (str, int)]:
    """One effective value plus enough provenance for an honest Admin UI."""

    value: Value
    source: SettingSource
    editable: bool


@dataclass(frozen=True, slots=True)
class WorkloadRuntimeConfiguration:
    """Effective mutable settings for one fixed API workload."""

    workload: str
    modal_app_name: EffectiveSetting[str]
    active_job_limit: EffectiveSetting[int]


@dataclass(frozen=True, slots=True)
class DatabaseOverridableSetting[Value: (str, int)]:
    """A static fallback that a same-transaction database read may replace."""

    value: Value
    database_override_allowed: bool

    def resolve(self, database_value: Value | None) -> Value:
        """Apply a database value unless an explicit process value wins."""
        if self.database_override_allowed and database_value is not None:
            return database_value
        return self.value


@dataclass(frozen=True, slots=True)
class JobAdmissionConfiguration:
    """Static configuration inputs resolved with SQLite during admission."""

    workload: str
    modal_environment: DatabaseOverridableSetting[str]
    modal_app_name: DatabaseOverridableSetting[str]
    workload_active_job_limit: DatabaseOverridableSetting[int]
    global_active_job_limit: DatabaseOverridableSetting[int]


@dataclass(frozen=True, slots=True)
class ModalConfigurationSnapshot:
    """Provider identity pinned to a Job when admission commits."""

    environment: str
    app_name: str


class RuntimeConfiguration:
    """Resolve process env over database over env file over defaults."""

    _MODAL_ENVIRONMENT_KEY = "modal_environment"
    _GLOBAL_ACTIVE_LIMIT_KEY = "global_active_job_limit"

    def __init__(self, store: ServiceStore, settings: ServiceSettings) -> None:
        """Bind live database overrides to immutable startup sources."""
        self.store = store
        self.settings = settings

    @property
    def modal_token_id(self) -> str:
        """Return the non-secret Modal service-user identifier."""
        if self.settings.modal_token_id is None:
            raise RuntimeError("Modal credentials were not validated at startup")
        return self.settings.modal_token_id

    @property
    def default_user_active_job_limit(self) -> int:
        """Return the limit copied onto newly provisioned Users."""
        return self.settings.default_user_active_job_limit

    def modal_environment(self) -> EffectiveSetting[str]:
        """Resolve the Modal Environment used by newly admitted Jobs."""
        return self._text_setting(
            environment_name="BIOMODALS_MODAL_ENVIRONMENT",
            database_key=self._MODAL_ENVIRONMENT_KEY,
            default=self.settings.modal_environment,
        )

    def global_active_job_limit(self) -> EffectiveSetting[int]:
        """Resolve the cross-workload non-terminal Job admission limit."""
        return self._integer_setting(
            environment_name="BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT",
            database_key=self._GLOBAL_ACTIVE_LIMIT_KEY,
            default=self.settings.global_active_job_limit,
        )

    def workload(self, workload: str) -> WorkloadRuntimeConfiguration:
        """Resolve settings for one fixed registered workload."""
        if workload != "gromacs":
            raise ValueError(f"Unknown workload: {workload}")
        stored = self.store.get_workload_configuration(workload)
        return WorkloadRuntimeConfiguration(
            workload=workload,
            modal_app_name=self._workload_text_setting(
                environment_name="BIOMODALS_GROMACS_APP",
                database_value=(stored.modal_app_name if stored is not None else None),
                default=self.settings.gromacs_app_name,
            ),
            active_job_limit=self._workload_integer_setting(
                environment_name="BIOMODALS_GROMACS_ACTIVE_LIMIT",
                database_value=(
                    stored.active_job_limit if stored is not None else None
                ),
                default=self.settings.gromacs_active_limit,
            ),
        )

    def admission_configuration(self, workload: str) -> JobAdmissionConfiguration:
        """Return static inputs; SQLite resolves mutable values atomically later."""
        if workload != "gromacs":
            raise ValueError(f"Unknown workload: {workload}")
        sources = self.settings.sources
        return JobAdmissionConfiguration(
            workload=workload,
            modal_environment=DatabaseOverridableSetting(
                self.settings.modal_environment,
                not sources.has_process_override("BIOMODALS_MODAL_ENVIRONMENT"),
            ),
            modal_app_name=DatabaseOverridableSetting(
                self.settings.gromacs_app_name,
                not sources.has_process_override("BIOMODALS_GROMACS_APP"),
            ),
            workload_active_job_limit=DatabaseOverridableSetting(
                self.settings.gromacs_active_limit,
                not sources.has_process_override("BIOMODALS_GROMACS_ACTIVE_LIMIT"),
            ),
            global_active_job_limit=DatabaseOverridableSetting(
                self.settings.global_active_job_limit,
                not sources.has_process_override("BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT"),
            ),
        )

    def update_environment(
        self,
        *,
        modal_environment: str | None | _Unchanged = _UNCHANGED,
        global_active_job_limit: int | None | _Unchanged = _UNCHANGED,
    ) -> None:
        """Atomically update supplied Environment fields after override checks."""
        updates: dict[str, str | None] = {}
        if not isinstance(modal_environment, _Unchanged):
            self._ensure_editable("BIOMODALS_MODAL_ENVIRONMENT")
            updates[self._MODAL_ENVIRONMENT_KEY] = (
                None
                if modal_environment is None
                else _nonempty(modal_environment, "Modal environment")
            )
        if not isinstance(global_active_job_limit, _Unchanged):
            self._ensure_editable("BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT")
            updates[self._GLOBAL_ACTIVE_LIMIT_KEY] = (
                None
                if global_active_job_limit is None
                else str(
                    _nonnegative(
                        global_active_job_limit,
                        "Global active job limit",
                    )
                )
            )
        self.store.set_service_settings(updates)

    def set_workload(
        self,
        workload: str,
        *,
        modal_app_name: str | None | _Unchanged = _UNCHANGED,
        active_job_limit: int | None | _Unchanged = _UNCHANGED,
    ) -> None:
        """Atomically update supplied settings for one fixed workload."""
        if workload != "gromacs":
            raise ValueError(f"Unknown workload: {workload}")
        updates: dict[str, str | int | None] = {}
        if not isinstance(modal_app_name, _Unchanged):
            self._ensure_editable("BIOMODALS_GROMACS_APP")
            updates["modal_app_name"] = (
                None
                if modal_app_name is None
                else _nonempty(modal_app_name, "Modal app name")
            )
        if not isinstance(active_job_limit, _Unchanged):
            self._ensure_editable("BIOMODALS_GROMACS_ACTIVE_LIMIT")
            updates["active_job_limit"] = (
                None
                if active_job_limit is None
                else _nonnegative(active_job_limit, "Tool active job limit")
            )
        self.store.set_workload_configuration(workload, updates)

    def _text_setting(
        self,
        *,
        environment_name: str,
        database_key: str,
        default: str,
    ) -> EffectiveSetting[str]:
        stored = self.store.get_service_setting(database_key)
        return self._setting(environment_name, stored, default, _nonempty)

    def _integer_setting(
        self,
        *,
        environment_name: str,
        database_key: str,
        default: int,
    ) -> EffectiveSetting[int]:
        stored = self.store.get_service_setting(database_key)
        return self._setting(
            environment_name,
            stored,
            default,
            _parse_nonnegative,
        )

    def _workload_text_setting(
        self,
        *,
        environment_name: str,
        database_value: str | None,
        default: str,
    ) -> EffectiveSetting[str]:
        return self._setting(environment_name, database_value, default, _nonempty)

    def _workload_integer_setting(
        self,
        *,
        environment_name: str,
        database_value: int | None,
        default: int,
    ) -> EffectiveSetting[int]:
        return self._setting(
            environment_name,
            database_value,
            default,
            _parse_nonnegative,
        )

    def _setting[Value: (str, int)](
        self,
        environment_name: str,
        database_value: object | None,
        default: Value,
        parse: Callable[[object, str], Value],
    ) -> EffectiveSetting[Value]:
        sources = self.settings.sources
        if sources.has_process_override(environment_name):
            raw_value = sources.process_environment[environment_name]
            return EffectiveSetting(
                parse(raw_value, environment_name),
                "process_environment",
                False,
            )
        if database_value is not None:
            return EffectiveSetting(
                parse(database_value, environment_name),
                "database",
                True,
            )
        file_value = sources.file_value(environment_name)
        if file_value is not None:
            return EffectiveSetting(
                parse(file_value, environment_name),
                "configuration_file",
                True,
            )
        return EffectiveSetting(default, "default", True)

    def _ensure_editable(self, environment_name: str) -> None:
        if self.settings.sources.has_process_override(environment_name):
            raise SettingOverrideError(
                f"{environment_name} is controlled by a process environment variable"
            )


def _nonempty(value: object, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _nonnegative(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be at least 0")
    return value


def _parse_nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label} must be an integer")
    return _nonnegative(int(value), label)
