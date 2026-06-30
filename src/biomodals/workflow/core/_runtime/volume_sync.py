"""Serialized Modal volume synchronization for workflow runtime state."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from biomodals.workflow.core.ledger import WorkflowLedger


class WorkflowVolume(Protocol):
    """Minimal Modal Volume boundary used by the workflow runtime."""

    def commit(self) -> object:
        """Persist pending writes to the mounted volume."""

    def reload(self) -> object:
        """Refresh local view of writes made by other containers."""


class WorkflowVolumeSync:
    """Coordinate Modal volume synchronization with ledger file access."""

    def __init__(
        self,
        *,
        workflow_volume: WorkflowVolume | None,
        ledger: WorkflowLedger,
    ) -> None:
        self.workflow_volume = workflow_volume
        self.ledger = ledger
        self.lock = RLock()

    def commit(self) -> None:
        """Persist pending workflow volume writes, if a Modal volume is attached."""
        if self.workflow_volume is None:
            return
        with self.lock:
            with self.ledger.closed_for_volume_sync():
                self.workflow_volume.commit()

    def reload(self) -> None:
        """Refresh the local workflow volume view, if a Modal volume is attached."""
        if self.workflow_volume is None:
            return
        with self.lock:
            with self.ledger.closed_for_volume_sync():
                self.workflow_volume.reload()
