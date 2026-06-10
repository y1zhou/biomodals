"""Shared private runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from biomodals.workflow.core._runtime.volume_sync import WorkflowVolumeSync
from biomodals.workflow.core.ledger import WorkflowLedger


@dataclass(frozen=True)
class RuntimeServices:
    """Shared dependencies for private workflow runtime collaborators."""

    ledger: WorkflowLedger
    volume_root: Path
    workflow_volume_name: str
    volume_sync: WorkflowVolumeSync
