"""Deferred app-owned artifact availability checks.

Workflow-volume artifact checks remain the default. This module reserves the
private runtime extension point for strict external checks that need a declared
Modal function with the required app-owned volumes mounted.
"""

from __future__ import annotations

from typing import NoReturn


def check_external_artifact_availability(*_: object, **__: object) -> NoReturn:
    """Placeholder for strict app-owned volume artifact verification."""
    raise NotImplementedError(
        "External app-owned artifact verification is deferred. Implement this "
        "with a declared Modal checker function that mounts the required app "
        "volumes before enabling strict external artifact checks."
    )
