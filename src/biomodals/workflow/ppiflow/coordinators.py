"""Operational concurrency configuration for the PPIFlow workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_CANDIDATE_CONCURRENCY = 4


def candidate_concurrency_from_config(
    *configs: Mapping[str, Any],
    default: int = DEFAULT_CANDIDATE_CONCURRENCY,
) -> int:
    """Resolve shared or per-stage candidate concurrency."""
    for config in configs:
        if "candidate_concurrency" in config:
            value = int(config["candidate_concurrency"])
            if value < 1:
                raise ValueError("candidate_concurrency must be at least 1")
            return value
    return default
