"""Configuration for the department-hosted Biomodals API service."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path


def _boolean(name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _positive_integer(name: str, default: int) -> int:
    value = int(environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _optional_positive_integer(name: str) -> int | None:
    raw_value = environ.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1 when configured")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """All host-specific settings for one API process."""

    state_dir: Path
    cache_dir: Path
    cache_max_bytes: int
    frontend_url: str
    allowed_origin: str
    secure_cookies: bool
    modal_environment: str
    gromacs_app_name: str
    gromacs_active_limit: int
    reconcile_interval_seconds: float
    intermediate_retention_days: int | None

    @classmethod
    def from_environment(cls) -> ServiceSettings:
        """Load settings from environment variables with local-safe defaults."""
        frontend_url = environ.get(
            "BIOMODALS_FRONTEND_URL", "http://localhost:5173"
        ).rstrip("/")
        allowed_origin = environ.get("BIOMODALS_ALLOWED_ORIGIN", frontend_url)
        state_dir = Path(environ.get("BIOMODALS_STATE_DIR", ".biomodals/state"))
        cache_dir = Path(environ.get("BIOMODALS_CACHE_DIR", ".biomodals/cache"))
        return cls(
            state_dir=state_dir,
            cache_dir=cache_dir,
            cache_max_bytes=_positive_integer(
                "BIOMODALS_CACHE_MAX_BYTES", 10 * 1024**3
            ),
            frontend_url=frontend_url,
            allowed_origin=allowed_origin.rstrip("/"),
            secure_cookies=_boolean("BIOMODALS_SECURE_COOKIES", False),
            modal_environment=environ.get("BIOMODALS_MODAL_ENVIRONMENT", "main"),
            gromacs_app_name=environ.get("BIOMODALS_GROMACS_APP", "Gromacs"),
            gromacs_active_limit=_positive_integer("BIOMODALS_GROMACS_ACTIVE_LIMIT", 2),
            reconcile_interval_seconds=_positive_float(
                "BIOMODALS_RECONCILE_SECONDS", 10
            ),
            intermediate_retention_days=_optional_positive_integer(
                "BIOMODALS_INTERMEDIATE_RETENTION_DAYS"
            ),
        )

    @property
    def database_path(self) -> Path:
        """Return the durable SQLite path, separate from the artifact cache."""
        return self.state_dir / "service.sqlite3"
