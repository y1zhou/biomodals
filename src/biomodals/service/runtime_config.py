"""Live non-secret service configuration with explicit source precedence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from biomodals.service.config import ServiceSettings
from biomodals.service.tools import ToolDefinition

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
class EffectiveSetting[Value: (str, int, bool)]:
    """One effective value plus enough provenance for an honest Admin UI."""

    value: Value
    source: SettingSource
    editable: bool


@dataclass(frozen=True, slots=True)
class ToolRuntimeConfiguration:
    """Effective mutable settings for one fixed API Tool."""

    tool: str
    modal_app_name: str
    modal_app_version: EffectiveSetting[int]
    active_job_limit: EffectiveSetting[int]
    job_logs_visible_to_owner: EffectiveSetting[bool]

    @property
    def max_active_provider_calls(self) -> int:
        """Derive the per-Job total container ceiling from Tool capacity."""
        return max(1, self.active_job_limit.value * 8)

    @property
    def max_active_gpu_provider_calls(self) -> int:
        """Derive the per-Job GPU container ceiling from Tool capacity."""
        multiplier = 5 if self.tool == "humanization" else 1
        return max(1, self.active_job_limit.value * multiplier)


@dataclass(frozen=True, slots=True)
class _ToolDefaults:
    modal_app_name: str
    modal_app_version: int
    active_job_limit: int


class RuntimeConfiguration:
    """Resolve process env over database over env file over defaults."""

    _MODAL_ENVIRONMENT_KEY = "modal_environment"
    _GLOBAL_ACTIVE_LIMIT_KEY = "global_active_job_limit"

    def __init__(
        self,
        store: ServiceStore,
        settings: ServiceSettings,
        *,
        tool_definitions: Sequence[ToolDefinition],
    ) -> None:
        """Bind live database overrides to immutable startup sources."""
        self.store = store
        self.settings = settings
        self._tool_definitions = {
            definition.key: definition for definition in tool_definitions
        }
        if len(self._tool_definitions) != len(tool_definitions):
            raise ValueError("Tool definition keys must be unique")
        sources = settings.sources
        self._tool_defaults = {
            definition.key: _ToolDefaults(
                modal_app_name=_nonempty(
                    sources.value(
                        definition.modal_app_name_environment,
                        definition.default_modal_app_name,
                    ),
                    definition.modal_app_name_environment,
                ),
                modal_app_version=_parse_positive(
                    sources.value(
                        definition.modal_app_version_environment,
                        str(definition.default_modal_app_version),
                    ),
                    definition.modal_app_version_environment,
                ),
                active_job_limit=_parse_nonnegative(
                    sources.value(
                        definition.active_job_limit_environment,
                        str(definition.default_active_job_limit),
                    ),
                    definition.active_job_limit_environment,
                ),
            )
            for definition in tool_definitions
        }

    def tool_definition(self, tool: str) -> ToolDefinition:
        """Return the static descriptor for one registered Tool."""
        try:
            return self._tool_definitions[tool]
        except KeyError as exc:
            raise ValueError(f"Unknown Tool: {tool}") from exc

    def tool_names(self) -> tuple[str, ...]:
        """Return Tool keys in their configured display order."""
        return tuple(self._tool_defaults)

    def _defaults(self, tool: str) -> _ToolDefaults:
        self.tool_definition(tool)
        return self._tool_defaults[tool]

    def modal_app_version_fallback(self, tool: str) -> int:
        """Return the startup version restored by a null Admin PATCH."""
        return self._defaults(tool).modal_app_version

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

    def tool(self, tool: str) -> ToolRuntimeConfiguration:
        """Resolve settings for one fixed registered Tool."""
        definition = self.tool_definition(tool)
        defaults = self._defaults(tool)
        stored = self.store.get_tool_configuration(tool)
        configuration = ToolRuntimeConfiguration(
            tool=tool,
            modal_app_name=defaults.modal_app_name,
            modal_app_version=self._tool_positive_integer_setting(
                environment_name=definition.modal_app_version_environment,
                database_value=(
                    stored.modal_app_version if stored is not None else None
                ),
                default=defaults.modal_app_version,
            ),
            active_job_limit=self._tool_integer_setting(
                environment_name=definition.active_job_limit_environment,
                database_value=(
                    stored.active_job_limit if stored is not None else None
                ),
                default=defaults.active_job_limit,
            ),
            job_logs_visible_to_owner=self._tool_boolean_setting(
                database_value=(
                    stored.job_logs_visible_to_owner if stored is not None else None
                ),
                default=definition.job_logs_visible_to_owner_default,
            ),
        )
        return configuration

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

    def set_tool(
        self,
        tool: str,
        *,
        modal_app_version: int | None | _Unchanged = _UNCHANGED,
        active_job_limit: int | None | _Unchanged = _UNCHANGED,
        job_logs_visible_to_owner: bool | None | _Unchanged = _UNCHANGED,
    ) -> None:
        """Atomically update supplied settings for one fixed Tool."""
        definition = self.tool_definition(tool)
        updates: dict[str, int | bool | None] = {}
        if not isinstance(modal_app_version, _Unchanged):
            self._ensure_editable(definition.modal_app_version_environment)
            updates["modal_app_version"] = (
                None
                if modal_app_version is None
                else _positive(modal_app_version, "Modal App version")
            )
        if not isinstance(active_job_limit, _Unchanged):
            self._ensure_editable(definition.active_job_limit_environment)
            updates["active_job_limit"] = (
                None
                if active_job_limit is None
                else _nonnegative(active_job_limit, "Tool active job limit")
            )
        if not isinstance(job_logs_visible_to_owner, _Unchanged):
            updates["job_logs_visible_to_owner"] = job_logs_visible_to_owner
        self.store.set_tool_configuration(tool, updates)

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

    def _tool_integer_setting(
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

    def _tool_positive_integer_setting(
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
            _parse_positive,
        )

    def _tool_boolean_setting(
        self,
        *,
        database_value: bool | None,
        default: bool,
    ) -> EffectiveSetting[bool]:
        """Resolve a database-editable boolean over its Tool-owned default."""
        if database_value is not None:
            return EffectiveSetting(database_value, "database", True)
        return EffectiveSetting(default, "default", True)

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


def _positive(value: int, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be at least 1")
    return value


def _parse_nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label} must be an integer")
    return _nonnegative(int(value), label)


def _parse_positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label} must be an integer")
    return _positive(int(value), label)
