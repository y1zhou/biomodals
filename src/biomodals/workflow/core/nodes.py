"""Base node contracts for Biomodals workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from biomodals.schema import (
    AppRunResult,
    NodeExecutionPolicy,
    NodePlacement,
    WorkflowArtifact,
)


@dataclass(frozen=True)
class NodeRunContext:
    """Runtime context passed to workflow node implementations."""

    run_id: str
    node_id: str
    attempt_id: str
    cache_dir: Path
    inputs: dict[str, list[WorkflowArtifact]]


class WorkflowNode(Protocol):
    """Protocol for one semantic workflow DAG vertex."""

    execution_policy: NodeExecutionPolicy
    placement: NodePlacement

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute node implementation and return app-compatible outputs."""


class WorkflowNativeNode:
    """Base class for workflow nodes implemented directly in workflow code."""

    execution_policy = NodeExecutionPolicy.RERUN
    placement = NodePlacement.ORCHESTRATOR

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute workflow-native logic."""
        raise NotImplementedError


class AppBackedNode:
    """Base class for workflow nodes implemented by calling app functions."""

    execution_policy = NodeExecutionPolicy.RERUN
    placement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute the app-backed node implementation."""
        raise NotImplementedError
