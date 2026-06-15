"""In-memory runtime diagnostics for scheduler decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from biomodals.workflow.core._runtime.scheduler import (
    SchedulerDecision,
    SchedulerDecisionStatus,
)


@dataclass(frozen=True)
class SchedulerDecisionDiagnostic:
    """Stable snapshot of one scheduler loop decision."""

    status: SchedulerDecisionStatus
    completed: tuple[str, ...]
    ready: tuple[str, ...] = ()
    running: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class RuntimeDiagnostics:
    """Runtime-visible diagnostics for the most recent workflow run."""

    run_id: str | None = None
    scheduler_decisions: list[SchedulerDecisionDiagnostic] = field(default_factory=list)

    @property
    def scheduled_waves(self) -> list[list[str]]:
        """Return node ids scheduled together in each ready scheduler pass."""
        return [
            list(decision.ready)
            for decision in self.scheduler_decisions
            if decision.status == SchedulerDecisionStatus.READY
        ]

    def record_scheduler_decision(self, decision: SchedulerDecision) -> None:
        """Record a stable snapshot of a scheduler decision."""
        self.scheduler_decisions.append(
            SchedulerDecisionDiagnostic(
                status=decision.status,
                completed=tuple(sorted(decision.completed)),
                ready=tuple(decision.ready),
                running=tuple(decision.running),
                blocked=tuple(decision.blocked),
                warnings=tuple(decision.warnings),
            )
        )
