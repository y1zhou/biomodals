"""Base node contracts for Biomodals workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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

    app_name: str | None = None
    function_name: str | None = None
    execution_policy = NodeExecutionPolicy.RERUN
    placement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Load and call the backing Modal function."""
        app_function = self.load_app_function()
        raw_result = self.invoke_app_function(
            app_function,
            self.build_app_function_kwargs(context),
        )
        return AppRunResult.model_validate(raw_result)

    def load_app_function(self) -> Any:
        """Load the backing Modal function lazily."""
        if self.app_name is None or self.function_name is None:
            raise NotImplementedError(
                "App-backed nodes must define app_name/function_name or override "
                "load_app_function()"
            )
        import modal

        return modal.Function.from_name(self.app_name, self.function_name)

    def build_app_function_kwargs(self, context: NodeRunContext) -> dict[str, Any]:
        """Build keyword arguments for the backing Modal function."""
        return {}

    def invoke_app_function(
        self,
        app_function: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Call a loaded Modal function."""
        if hasattr(app_function, "remote"):
            return app_function.remote(**kwargs)
        return app_function(**kwargs)
