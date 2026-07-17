"""Configuration for the department-hosted Biomodals API service."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values


@dataclass(frozen=True, slots=True)
class ConfigurationSources:
    """Keep process and file values separate so precedence stays observable."""

    process_environment: Mapping[str, str] = field(repr=False)
    file_environment: Mapping[str, str] = field(repr=False)

    def has_process_override(self, name: str) -> bool:
        """Return whether the process explicitly supplied one value."""
        return name in self.process_environment

    def file_value(self, name: str) -> str | None:
        """Return one configuration-file value without consulting the process."""
        return self.file_environment.get(name)

    def value(self, name: str, default: str) -> str:
        """Resolve process environment over configuration file over default."""
        if name in self.process_environment:
            return self.process_environment[name]
        return self.file_environment.get(name, default)


def _boolean(sources: ConfigurationSources, name: str, default: bool) -> bool:
    value = sources.value(name, str(default))
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _positive_integer(
    sources: ConfigurationSources,
    name: str,
    default: int,
) -> int:
    value = int(sources.value(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _optional_positive_integer(
    sources: ConfigurationSources,
    name: str,
) -> int | None:
    raw_value = sources.value(name, "")
    if not raw_value.strip():
        return None
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1 when configured")
    return value


def _positive_float(
    sources: ConfigurationSources,
    name: str,
    default: float,
) -> float:
    value = float(sources.value(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _required_text(sources: ConfigurationSources, name: str, default: str) -> str:
    value = sources.value(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _public_url(sources: ConfigurationSources) -> str:
    value = _required_text(
        sources,
        "BIOMODALS_PUBLIC_URL",
        "http://localhost:5173",
    ).rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("BIOMODALS_PUBLIC_URL must be an HTTP(S) origin")
    return value


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """All host-specific settings and lower-precedence runtime defaults."""

    sources: ConfigurationSources = field(repr=False)
    state_dir: Path
    cache_dir: Path
    cache_max_bytes: int
    public_url: str
    secure_cookies: bool
    modal_environment: str
    gromacs_app_name: str
    gromacs_active_limit: int
    global_active_job_limit: int
    default_user_active_job_limit: int
    reconcile_interval_seconds: float
    intermediate_retention_days: int | None
    modal_token_id: str | None
    modal_token_secret: str | None = field(repr=False)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> ServiceSettings:
        """Load an explicit env file, then overlay the process environment."""
        process_environment = dict(os.environ if environment is None else environment)
        file_environment: dict[str, str] = {}
        configured_path = process_environment.get("BIOMODALS_API_CONF_ENV", "").strip()
        if configured_path:
            path = Path(configured_path).expanduser()
            if not path.is_file():
                raise ValueError(f"BIOMODALS_API_CONF_ENV file does not exist: {path}")
            file_environment = {
                key: value
                for key, value in dotenv_values(path).items()
                if value is not None
            }
        sources = ConfigurationSources(process_environment, file_environment)
        return cls(
            sources=sources,
            state_dir=Path(sources.value("BIOMODALS_STATE_DIR", ".biomodals/state")),
            cache_dir=Path(sources.value("BIOMODALS_CACHE_DIR", ".biomodals/cache")),
            cache_max_bytes=_positive_integer(
                sources,
                "BIOMODALS_CACHE_MAX_BYTES",
                10 * 1024**3,
            ),
            public_url=_public_url(sources),
            secure_cookies=_boolean(sources, "BIOMODALS_SECURE_COOKIES", False),
            modal_environment=_required_text(
                sources,
                "BIOMODALS_MODAL_ENVIRONMENT",
                "production",
            ),
            gromacs_app_name=_required_text(
                sources,
                "BIOMODALS_GROMACS_APP",
                "Gromacs",
            ),
            gromacs_active_limit=_positive_integer(
                sources,
                "BIOMODALS_GROMACS_ACTIVE_LIMIT",
                2,
            ),
            global_active_job_limit=_positive_integer(
                sources,
                "BIOMODALS_GLOBAL_ACTIVE_JOB_LIMIT",
                10,
            ),
            default_user_active_job_limit=_positive_integer(
                sources,
                "BIOMODALS_DEFAULT_USER_ACTIVE_JOB_LIMIT",
                2,
            ),
            reconcile_interval_seconds=_positive_float(
                sources,
                "BIOMODALS_RECONCILE_SECONDS",
                10,
            ),
            intermediate_retention_days=_optional_positive_integer(
                sources,
                "BIOMODALS_INTERMEDIATE_RETENTION_DAYS",
            ),
            modal_token_id=sources.value("MODAL_TOKEN_ID", "").strip() or None,
            modal_token_secret=(
                sources.value("MODAL_TOKEN_SECRET", "").strip() or None
            ),
        )

    @property
    def database_path(self) -> Path:
        """Return the durable SQLite path, separate from the artifact cache."""
        return self.state_dir / "service.sqlite3"

    def require_modal_credentials(self) -> None:
        """Fail backend startup unless both service-user credentials exist."""
        if self.modal_token_id is None or self.modal_token_secret is None:
            raise ValueError(
                "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set to start the API"
            )

    def install_modal_credentials(self) -> None:
        """Expose file-sourced credentials to the Modal SDK without overriding env."""
        token_id = self.modal_token_id
        token_secret = self.modal_token_secret
        if token_id is None or token_secret is None:
            self.require_modal_credentials()
            raise RuntimeError("Modal credential validation returned unexpectedly")
        os.environ.setdefault("MODAL_TOKEN_ID", token_id)
        os.environ.setdefault("MODAL_TOKEN_SECRET", token_secret)
