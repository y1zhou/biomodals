"""Modal function lookup, detached submission, polling, and cancellation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import modal

from biomodals.service.gromacs.plan import (
    REQUIRED_FUNCTIONS,
    modal_invocation,
    prepare_operation,
)
from biomodals.service.gromacs.router import GromacsJobOptions
from biomodals.service.runtime_config import ModalConfigurationSnapshot
from biomodals.service.store import JobRecord
from biomodals.service.submission import SubmissionOutcomeUnknownError

LOGGER = logging.getLogger(__name__)
MODAL_SERVICE_ERRORS = (
    modal.exception.AuthError,
    modal.exception.ConnectionError,
    modal.exception.InternalError,
    modal.exception.InvalidError,
    modal.exception.NotFoundError,
    modal.exception.PermissionDeniedError,
    modal.exception.ResourceExhaustedError,
    modal.exception.ServiceError,
)
DEFINITE_SUBMISSION_ERRORS = (
    modal.exception.AuthError,
    modal.exception.InvalidError,
    modal.exception.NotFoundError,
    modal.exception.PermissionDeniedError,
)


@dataclass(frozen=True, slots=True)
class SubmittedModalCall:
    """Identifiers persisted after detached submission."""

    modal_call_id: str
    run_name: str
    operation: str


@dataclass(frozen=True, slots=True)
class PollOutcome:
    """Sanitized state observed from one Modal call graph."""

    kind: Literal["running", "completed", "cancelled", "failed", "expired"]


def _call_nodes(
    roots: Iterable[modal.call_graph.InputInfo],
) -> list[modal.call_graph.InputInfo]:
    nodes: list[modal.call_graph.InputInfo] = []
    pending = list(roots)
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(node.children)
    return nodes


class ModalGromacsProvider:
    """Invoke one separately deployed GROMACS App through the Modal SDK."""

    def __init__(
        self,
        *,
        output_volume_name: str = "Gromacs-outputs",
        call_resolver: Callable[[str], modal.FunctionCall] = modal.FunctionCall.from_id,
        function_resolver: Callable[..., modal.Function] | None = None,
    ) -> None:
        """Configure deployed-function resolution and required output Volume."""
        self.output_volume_name = output_volume_name
        self._function_resolver = function_resolver or modal.Function.from_name
        self._call_resolver = call_resolver

    async def preflight(
        self,
        app_name: str,
        environment_name: str,
        app_version: int,
    ) -> None:
        """Hydrate required deployed resources without invoking compute."""
        volume = modal.Volume.from_name(
            self.output_volume_name,
            environment_name=environment_name,
        )
        await volume.hydrate.aio()
        for function_name in REQUIRED_FUNCTIONS:
            function = self._function_resolver(
                app_name,
                function_name,
                environment_name=environment_name,
                version=app_version,
            )
            await function.hydrate.aio()

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
        *,
        run_name: str,
        modal_configuration: ModalConfigurationSnapshot,
    ) -> SubmittedModalCall:
        """Spawn the first deployed compute stage without a remote coordinator."""
        function_name = prepare_operation(cpu_only=options.cpu_only)
        function = self._function_resolver(
            modal_configuration.app_name,
            function_name,
            environment_name=modal_configuration.environment,
            version=modal_configuration.app_version,
        )
        try:
            call = await function.spawn.aio(
                pdb_content=pdb_content,
                run_name=run_name,
                simulation_time_ns=options.simulation_time_ns,
                run_pdbfixer=options.run_pdbfixer,
            )
            modal_call_id = call.object_id
        except DEFINITE_SUBMISSION_ERRORS:
            raise
        except Exception as exc:
            raise SubmissionOutcomeUnknownError(
                "Modal did not return a durable FunctionCall handle"
            ) from exc
        return SubmittedModalCall(
            modal_call_id=modal_call_id,
            run_name=run_name,
            operation=function_name,
        )

    async def submit_operation(
        self,
        job: JobRecord,
        operation: str,
    ) -> SubmittedModalCall:
        """Spawn one explicitly selected deployed stage in the Job graph."""
        if job.run_name is None:
            raise ValueError("GROMACS Job has no run name")
        options = GromacsJobOptions.model_validate_json(job.parameters_json)
        invocation = modal_invocation(
            operation,
            cpu_only=options.cpu_only,
            run_name=job.run_name,
            simulation_time_ns=options.simulation_time_ns,
        )

        function = self._function_resolver(
            job.modal_app_name,
            invocation.function_name,
            environment_name=job.modal_environment,
            version=job.modal_app_version,
        )
        try:
            call = await function.spawn.aio(**invocation.kwargs)
            modal_call_id = call.object_id
        except DEFINITE_SUBMISSION_ERRORS:
            raise
        except Exception as exc:
            raise SubmissionOutcomeUnknownError(
                "Modal did not return a durable FunctionCall handle"
            ) from exc
        return SubmittedModalCall(
            modal_call_id=modal_call_id,
            run_name=job.run_name,
            operation=operation,
        )

    async def poll(
        self,
        modal_call_id: str,
        *,
        operation: str | None = None,
    ) -> PollOutcome:
        """Poll without blocking and distinguish a poll timeout from failure."""
        call = self._call_resolver(modal_call_id)
        try:
            raw_result = await call.get.aio(timeout=0)
        except modal.exception.InputCancellation:
            graph = await call.get_call_graph.aio()
            nodes = _call_nodes(graph)
            if any(
                node.status == modal.call_graph.InputStatus.PENDING for node in nodes
            ):
                return PollOutcome("running")
            return PollOutcome("cancelled")
        except (modal.exception.OutputExpiredError, modal.exception.NotFoundError):
            return PollOutcome("expired")
        except TimeoutError:
            graph = await call.get_call_graph.aio()
            nodes = _call_nodes(graph)
            if not nodes or any(
                node.status == modal.call_graph.InputStatus.PENDING for node in nodes
            ):
                return PollOutcome("running")
            if any(
                node.status == modal.call_graph.InputStatus.TERMINATED for node in nodes
            ):
                return PollOutcome("cancelled")
            if any(
                node.status
                in {
                    modal.call_graph.InputStatus.FAILURE,
                    modal.call_graph.InputStatus.INIT_FAILURE,
                    modal.call_graph.InputStatus.TIMEOUT,
                }
                for node in nodes
            ):
                return PollOutcome("failed")
            # A SUCCESS graph can become visible just before get() can return
            # retained output. Poll again rather than inventing a failure.
            return PollOutcome("running")
        except MODAL_SERVICE_ERRORS:
            raise
        except Exception:
            return PollOutcome("failed")
        if not isinstance(raw_result, str):
            LOGGER.error(
                "GROMACS stage %s (%s) returned an invalid result",
                operation,
                modal_call_id,
            )
            return PollOutcome("failed")
        return PollOutcome("completed")

    async def cancel(self, modal_call_id: str) -> None:
        """Cancel the root and every currently visible active descendant."""
        root = self._call_resolver(modal_call_id)
        graph = await root.get_call_graph.aio()
        nodes = _call_nodes(graph)
        if nodes and not any(
            node.status == modal.call_graph.InputStatus.PENDING for node in nodes
        ):
            return
        call_ids = {
            node.function_call_id
            for node in nodes
            if node.status == modal.call_graph.InputStatus.PENDING
            and node.function_call_id
        }
        call_ids.discard(modal_call_id)
        for call_id in sorted(call_ids):
            call = self._call_resolver(call_id)
            await call.cancel.aio(terminate_containers=False)
        await root.cancel.aio(terminate_containers=False)
